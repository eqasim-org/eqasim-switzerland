import React, { useState, useEffect } from "react";
import "./ChoroplethControls.css";

// Define mode colors for the legend
const MODE_COLORS = {
  car: "#636efa",
  car_passenger: "#ef553b",
  pt: "#00cc96",
  bike: "#ab63fa",
  walk: "#ffa15a",
};

const ChoroplethControls = ({ selectedMode, setSelectedMode, selectedDataset, setSelectedDataset, updateMapSymbology }) => {
  const [maxSharePerMode, setMaxSharePerMode] = useState(null);

  // Fetch max shares per mode for legend scaling
  useEffect(() => {
    fetch("/data/canton_mode_share.json")
      .then((response) => response.json())
      .then((data) => setMaxSharePerMode(data.max_share_per_mode))
      .catch((error) => console.error("Error loading max share per mode:", error));
  }, []);

  // Handle mode selection
  const handleModeChange = (e) => {
    const newMode = e.target.value;
    setSelectedMode(newMode);
    updateMapSymbology(newMode, selectedDataset);
  };

  // Handle dataset toggle
  const handleDatasetChange = () => {
    const newDataset = selectedDataset === "Microcensus" ? "Synthetic" : "Microcensus";
    setSelectedDataset(newDataset);
    updateMapSymbology(selectedMode, newDataset);
  };

  return (
    <div className="choropleth-controls">
      <label>Select Mode:</label>
      <select value={selectedMode} onChange={handleModeChange}>
        <option value="None">None</option>
        <option value="car">Car</option>
        <option value="car_passenger">Car Passenger</option>
        <option value="bike">Bike</option>
        <option value="pt">Public Transport</option>
        <option value="walk">Walking</option>
      </select>

      <div className="dataset-toggle">
        <span>Microcensus</span>
        <label className="switch">
          <input type="checkbox" checked={selectedDataset === "Synthetic"} onChange={handleDatasetChange} />
          <span className="slider round"></span>
        </label>
        <span>Synthetic</span>
      </div>

      {/* Legend */}
      {selectedMode !== "None" && maxSharePerMode && (
        <div className="legend">
          <h4>Legend</h4>
          <div className="legend-container">
            <span className="legend-label">0%</span>
            <div className="legend-gradient" style={{
              background: `linear-gradient(to left, ${MODE_COLORS[selectedMode]} 0%, #FFFFFF 100%)`
            }}></div>
            <span className="legend-label">{Math.round(maxSharePerMode[selectedMode] * 100)}%</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default ChoroplethControls;
