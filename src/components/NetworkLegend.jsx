import React from "react";
import "./NetworkLegend.css";

const Legend = ({ selectedGraph }) => {
  const isVolumes = selectedGraph === "Volumes";
  const isNetwork = selectedGraph === "Network";

  if (!isVolumes && !isNetwork) return null;

  return (
    <div className="network-legend-container">
      {/* Speed/Volume Legend */}
      <div className="network-legend-section">
        <div className="network-legend-title">
          {isVolumes ? "Average Daily Volume [vehicles/day]" : "Network Speed [km/h]"}
        </div>
        <div className="network-legend-bar">
          {isVolumes ? (
            <>
              <span className="network-legend-label">0</span>
              <span className="network-legend-label">100</span>
              <span className="network-legend-label">500</span>
            </>
          ) : (
            <>
              <span className="network-legend-label">0</span>
              <span className="network-legend-label">50</span>
              <span className="network-legend-label">100</span>
              <span className="network-legend-label">150</span>
            </>
          )}
        </div>
        <div
          className="network-legend-gradient"
          style={{
            background: isVolumes
              ? "linear-gradient(to right, #ffffcc, #c2e699, #78c679, #31a354, #006837)"
              : "linear-gradient(to right, #ffffb2, #fed976, #feb24c, #fd8d3c, #fc4e2a, #e31a1c, #b10026)",
          }}
        />
      </div>

      {/* Capacity Legend */}
      <div className="network-legend-section">
        <div className="network-legend-title">Road Capacity</div>
        <div className="capacity-legend">
          <div className="capacity-item">
            <div className="capacity-line thin"></div>
            <span className="network-legend-label">300</span>
          </div>
          <div className="capacity-item">
            <div className="capacity-line medium"></div>
            <span className="network-legend-label">2000</span>
          </div>
          <div className="capacity-item">
            <div className="capacity-line thick"></div>
            <span className="network-legend-label">4000+</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Legend;
