# Mode Choice Module

## 1. Introduction

This presentation provides an overview of the **Mode Choice Module**.

### Goal

The primary goal of this module is to assign a transportation mode to each trip in the synthetic population. The assignment is based on individual agent characteristics (e.g., age, income, car availability) and trip-specific details (e.g., travel time, origin, destination, purpose).

### Key Capabilities

The module supports three main operations:

1.  **Estimation**: Estimate discrete mode choice model parameters from observed survey data (Swiss Microcensus) using the Biogeme library.
2.  **Calibration**: Fine-tune the Alternative-Specific Constants (ASCs) to match desired aggregate mode shares in the target population.
3.  **Prediction**: Apply the calibrated model to predict the transportation mode for each agent's trip.

---

## 2. Architecture Overview

The module is organized into several sub-components within the `mode_choice/` directory. Each component has a specific responsibility:

| Component | Path | Description |
|---|---|---|
| **Entry Point** | `run.py` | Orchestrates the entire mode choice process. |
| **Data Preparation** | `prepare_data.py` | Loads and formats input data (persons, tours, variables). |
| **Tour Building** | `tours/` | Constructs activity-based tours from individual trips. |
| **Variables** | `variables/` | Computes the variables for each mode (e.g., travel time, distance). |
| **Car Routing** | `network/` | Handles car routing on the road network, with or without congestion. |
| **Costs** | `cost/` | Calculates monetary costs (PT fares, car fuel, parking). |
| **Core Logic (DMC)** | `dmc/` | Contains the Discrete Mode Choice model implementation. |
| **Estimation** | `estimate_model/` | Estimates model parameters from Microcensus data using Biogeme. |
| **Calibration** | `calibration/` | Calibrates the model to match target mode shares. |

---

## 3. Workflow

The execution flow is managed by `mode_choice/run.py` and proceeds through the following stages:

### Step 1: Configuration
The module reads its settings from `config.yml`. This file contains default configuration values (e.g., data paths, DMC parameters, flags for estimation/calibration) that control the module's behavior. These configurations can be modified from this file or `mode_choice.dmc_defaults` which contains the default values of all the parameters related to mode choice.

### Step 2: Data Preparation
The `prepare_data` stage loads and formats the necessary inputs:
-   **Persons**: Socio-demographic attributes (age, sex, income, driving license).
-   **Tours/Trips**: Travel demand information (origin, destination, purpose) and the difference possible tours that can be realized (i.e. car-car-car, pt-walk-pt, pt-pt-pt, ...).
-   **Variables**: variables for each trips and each mode (travel times, distances, costs).

### Step 3: Parameter Handling
The module determines which parameters to use based on the configuration flags:
-   If **Estimation** is enabled → Runs the `estimate_model` stage.
-   If **Calibration** is enabled → Runs the `calibration` stage (optionally after estimation).
-   Otherwise → Loads pre-existing parameters from `dmc_parameters.yml`.

### Step 4: Model Execution
The `DMC` class is initialized with the chosen parameters and prepared data. It then calculates utilities for each mode and selects the final mode for each tour.

### Step 5: Output
The module returns the predicted mode for each trip (trips with 0 distance are filtered out in an earlier stage), ready for integration into the synthetic population.

---

## 4. Core Component: DMC Engine

The core prediction logic resides in `mode_choice/dmc/`. It is structured as follows:

### DMC Class (`run_dmc.py`)
-   The main controller for the mode choice process.
-   Initializes with parameters, tour data, person attributes, and modes variables.
-   Orchestrates utility calculation and mode selection.

### Utilities (`utilities/`)
This sub-module calculates the utility of each transportation mode:
-   **`TourUtility`**: Aggregates utilities across all trips within a tour.
-   **Mode-Specific Utilities**: Dedicated classes for each mode (`CarUtility`, `PtUtility`, `BikeUtility`, `WalkUtility`, `CpUtility` for car passenger).
-   **`Parameters`**: Manages loading and accessing model coefficients from the YAML file.

### Selector (`selector/`)
-   **`Selector`**: Implements the probabilistic mode selection based on a Multinomial Logit model. It converts utilities into choice probabilities and draws a mode for each tour. A seed parameter is used for reproducibility. For efficient selection (fast), the selection is done this way: first, a random error is sampled from a Gumbel distribution, this error is added to the deterministic utility, and then maximum utility selection is done, meaning the choice with the highest global utility is selected.

---

## 5. Estimation & Calibration

### Estimation (`estimate_model/`)

-   **Tool**: The Biogeme library for discrete choice modeling.
-   **Data**: Swiss Microcensus (observed travel behavior).
-   **Process**: Estimates utility function coefficients (betas) using maximum likelihood estimation.
-   **Output**: A YAML file containing the estimated parameters.
-   **Consistency**: For consistency with MATSim data format, when writing the parameters, a renaming is done.

### Calibration (`calibration/`)

-   **Goal**: Adjust the Alternative-Specific Constants (ASCs) so that the simulated mode shares match a set of target shares that are estimated in `data.microcensus.shares`.
-   **Process**:
    1.  Run the DMC model with initial parameters.
    2.  Compare simulated mode shares against target shares.
    3.  Use an optimizer to iteratively adjust the ASCs until convergence.
-   **Output**: A `calibrated_parameters.yml` file with the final, calibrated parameters.
-  **Why ?**: The utility function parameters are estimated using Microcensus data, which represent only a sample of the population and require, and further filtered to ensure data quality for model estimation. As a result, the estimation sample may not be fully representative of the entire population. Moreover, the explanatory variables included in the utility specifications cannot capture all relevant determinants of mode choice, such as comfort, perceived safety, environmental awareness, physical ability, or other latent preferences. The Alternative-Specific Constants (ASCs) absorb the average effect of these unobserved or omitted factors. When estimated from a restricted sample, ASCs may therefore be biased. The calibration procedure corrects for this bias by adjusting the ASCs so that the model reproduces observed aggregate mode shares, ensuring that the simulated choices are consistent with population-level behavior rather than solely reflecting the estimation sample.

---

## 6. Configuration Options

The module is controlled via `config.yml`. The extra key settings include:

| Option | Description |
|---|---|
| `estimate_dmc_parameters` | Set to `true` to run the estimation stage. |
| `calibrate_dmc_parameters` | Set to `true` to run the calibration stage. |
| `mode_parameters_path` | Path to a pre-existing parameter file (used if not estimating). |
| `random_seed` | Ensures reproducibility of results. |

---

## 7. Summary

The **Mode Choice Module** is a robust, modular, and configurable system that:

1.  **Prepares** complex data inputs from multiple sources.
2.  **Estimates** model parameters from real-world survey data.
3.  **Calibrates** the model to match target mode shares.
4.  **Predicts** transportation mode choices using a state-of-the-art Discrete Mode Choice framework.
5.  **Integrates** seamlessly into the broader synthetic population pipeline.

## 8. Possible issues
1. If you encounter an OOM (Out of Memory) error, it means the tours DataFrame is too large. You might want to perform mode choice in batches. To do this, specify `num_tour_batches` in the config file as a high number based on your memory size (10, 50, ...). This will divide the tours into that number of chunks and process mode choice batch by batch in series. It is already efficient, but do not attempt to parallelize this part, because it cannot be made parallel due to the code's organization. Polars already handles parallelism internally when performing its operations.