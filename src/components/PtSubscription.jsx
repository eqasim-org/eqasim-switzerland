import React from "react";
import GenericBarPlot from "./GenericBarPlot";
import BasicBarPlot from "./BasicBarPlot";

const AGE_VARIABLES = {
  "[6, 15)": "[6, 15)",
  "[15, 18)": "[15, 18)",
  "[18, 24)": "[18, 24)",
  "[24, 30)": "[24, 30)",
  "[30, 45)": "[30, 45)",
  "[45, 65)": "[45, 65)",
  "[65, 80)": "[65, 80)",
};

const GENDER_VARIABLES = {
  "Male": "0",
  "Female": "1",
};

const INCOME_VARIABLES = {
  "1": "1",
  "2": "2",
  "3": "3",
  "4": "4",
  "5": "5",
  "6": "6",
  "7": "7",
  "8": "8",
};

const PtSubscription = ({ canton, onClose }) => {
  return (
    <div className="overlay-panel">
      <h3>{canton || "All"} - Public Transport Subscription</h3>

      <BasicBarPlot
        dataFile="/data/pt_subscriptions.json"
        title="Public Transport Subscription Distribution"
        xAxisTitle="Subscription Type"
        yAxisTitle="Proportion"
        canton={canton}
      />

      <GenericBarPlot
        dataFile="/data/pt_sub_age.json"
        title="Public Transport Subscriptions by Age"
        xAxisTitle="Subscription Type"
        variables={AGE_VARIABLES}
        defaultVariable={AGE_VARIABLES["[6, 15)"]}
        canton={canton}
        onClose={onClose}
      />

      <GenericBarPlot
        dataFile="/data/pt_sub_gender.json"
        title="Public Transport Subscriptions by Gender"
        xAxisTitle="Subscription Type"
        variables={GENDER_VARIABLES}
        defaultVariable={GENDER_VARIABLES["Female"]}
        canton={canton}
        onClose={onClose}
      />

      <GenericBarPlot
        dataFile="/data/pt_sub_income.json"
        title="Public Transport Subscriptions by Income"
        xAxisTitle="Subscription Type"
        variables={INCOME_VARIABLES}
        defaultVariable={INCOME_VARIABLES["1"]}
        canton={canton}
        onClose={onClose}
      />
    </div>
  );
};

export default PtSubscription;
