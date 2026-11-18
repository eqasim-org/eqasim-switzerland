"""Experimental count and attribute-reconstruction heads, confined to analysis."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from synthesis.population.matching.V2.model import EmbeddingGRUModel
from synthesis.population.matching.V2.losses import MatchingLoss
from synthesis.population.matching.V2.features import CONTINUOUS

COUNT_LABELS = ('total', 'shop', 'leisure', 'education', 'work')
RECON_CONT = ('age', 'income_class', 'household_size', 'N_children_under_3',
              'N_children_under_6', 'N_children_under_12', 'N_children_under_18',
              'N_adults', 'population_density_log', 'commute_distance_log')
RECON_CAT = ('sex', 'car_availability', 'number_of_cars_class', 'driving_license',
             'ovgk', 'employed', 'is_student', 'employment_status',
             'work_location_type', 'municipality_type')


class AblationModel(EmbeddingGRUModel):
    def __init__(self, *args, count_head=False, explicit_reconstruction=False,
                 missing_categories=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.count_head = count_head
        self.explicit_reconstruction = explicit_reconstruction
        self.missing_categories = missing_categories or {}
        if count_head:
            # One shared head, five outputs: total, shopping, leisure, education, work.
            self.trip_count_head = nn.Sequential(nn.Linear(16, 8), nn.LayerNorm(8), nn.SiLU(),
                                                  nn.Linear(8, 5), nn.Softplus())
        if explicit_reconstruction:
            self.attribute_numeric = nn.Linear(16, len(RECON_CONT))
            self.attribute_categories = nn.ModuleDict({k: nn.Linear(16, self.card[k]) for k in RECON_CAT})

    def forward(self, x, seq_len):
        result = super().forward(x, seq_len)
        if not self.count_head:
            result['trip_count'] = result['trip_count'].reshape(-1)
        if self.explicit_reconstruction:
            z = result['latent']
            prediction = self.attribute_numeric(z)
            total, mass = z.new_zeros(len(z)), z.new_zeros(len(z))
            for j, key in enumerate(RECON_CONT):
                index = CONTINUOUS.index(key)
                valid = x['cont'][:, len(CONTINUOUS) + index].eq(0)
                error = F.smooth_l1_loss(prediction[:, j], x['cont'][:, index], reduction='none')
                total = total + error * valid
                mass = mass + valid
            for key in RECON_CAT:
                valid = x[key].ne(0) & x[key].ne(self.missing_categories.get(key, -1))
                error = F.cross_entropy(self.attribute_categories[key](z), x[key], reduction='none')
                total = total + error * valid
                mass = mass + valid
            # Replace projected-input reconstruction, keeping its existing modest weight.
            result['reconstruction'] = total / mass.clamp(min=1)
        return result


class AblationLoss(MatchingLoss):
    @staticmethod
    def trip_count(prediction, target):
        if prediction.ndim == 1:
            return MatchingLoss.trip_count(prediction, target[:, 0])
        error = F.smooth_l1_loss(prediction, target, reduction='none')
        # Preserve the original total-trip term and give the four purpose terms
        # together the same weight, rather than multiplying count influence by five.
        return error[:, 0] + error[:, 1:].mean(1)

    @staticmethod
    def reconstruction(prediction, encoder_input):
        return prediction if prediction.ndim == 1 else MatchingLoss.reconstruction(prediction, encoder_input)


def count_targets(source, trips, scale):
    """Count complete diaries, including secondary purposes and beyond truncation."""
    frame = trips[['person_id', 'purpose']].copy()
    frame['purpose'] = frame.purpose.astype(str).str.replace('_secondary', '', regex=False)
    grouped = frame.groupby(['person_id', 'purpose']).size().unstack(fill_value=0)
    counts = np.zeros((len(source), 5), dtype=np.float32)
    counts[:, 0] = frame.groupby('person_id').size().reindex(source.mz_id, fill_value=0)
    for j, key in enumerate(COUNT_LABELS[1:], 1):
        if key in grouped:
            counts[:, j] = grouped[key].reindex(source.mz_id, fill_value=0)
    return np.log1p(counts) / np.log1p(scale)
