import React, { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import GenericBarPlot from "./GenericBarPlot";

const DATASET_COLORS = {
  Microcensus: "#4A90E2",
  Synthetic: "#E07A5F",
};

const VARIABLES = {
    "Education": "education",
    "Work": "work",
    "Other": "other",
    "Shop": "shop",
    "Leisure": "leisure",
};

const DepartureTimes = ({ canton, onClose }) => {
  return (
    <div className="overlay-panel">
      <h3>{canton || "All"} - Departure Times</h3>
      
      <GenericBarPlot
        dataFile="/data/departure_times.json"
        title="Depature Times by Activity"
        xAxisTitle="Departure Time [HH:mm:ss]"
        variables={VARIABLES}
        defaultVariable={VARIABLES["Work"]}
        canton={canton}
        onClose={onClose}
      > </GenericBarPlot>
    </div>
  );
};

export default DepartureTimes