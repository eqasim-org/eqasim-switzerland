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
  Male: "0",
  Female: "1",
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

const CarAvailability = ({ canton, onClose }) => {
  return (
    <div className="overlay-panel">
      <h3>{canton || "Zurich"} - Car Availability</h3>

      <BasicBarPlot
        dataFile="/data/car_availability.json"
        title="Car Availability Class Distribution"
        xAxisTitle="Car Class"
        yAxisTitle="Proportion"
        canton={canton}
      />

      <GenericBarPlot
        dataFile="/data/num_cars_age.json"
        title="Car Availability Class by Age"
        xAxisTitle="Car Availability Class"
        variables={AGE_VARIABLES}
        defaultVariable={AGE_VARIABLES["[6, 15)"]}
        canton={canton}
        onClose={onClose}
      />

      <GenericBarPlot
        dataFile="/data/num_cars_gender.json"
        title="Car Availability Class by Gender"
        xAxisTitle="Car Availability Class"
        variables={GENDER_VARIABLES}
        defaultVariable={GENDER_VARIABLES["Female"]}
        canton={canton}
        onClose={onClose}
      />

      <GenericBarPlot
        dataFile="/data/num_cars_income.json"
        title="Car Availability Class by Income"
        xAxisTitle="Car Availability Class"
        variables={INCOME_VARIABLES}
        defaultVariable={INCOME_VARIABLES["1"]}
        canton={canton}
        onClose={onClose}
      />
    </div>
  );
};

export default CarAvailability;
