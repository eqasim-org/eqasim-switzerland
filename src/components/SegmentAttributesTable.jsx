import React from "react";
import "./Table.css"; // Reuse the same CSS

const format = (value, isSpeed = false) => {
  if (isSpeed) return `${Math.round(value * 3.6)} km/h`; // Convert m/s → km/h
  return `${Math.round(parseFloat(value))}`;
};

const SegmentAttributesTable = ({ properties }) => {
  if (!properties) return null;

  const { length, freespeed, capacity, permlanes, modes } = properties;
  const modeList = modes?.split(",") || [];

  return (
    <div className="canton-mode-share">
      <h4>Segment Info</h4>
      <table>
        <tbody>
          <tr><td>Length</td><td>{format(length)} m</td></tr>
          <tr><td>Free Speed</td><td>{format(freespeed, true)}</td></tr>
          <tr><td>Capacity</td><td>{format(capacity)}</td></tr>
          <tr><td>Lanes</td><td>{format(permlanes)}</td></tr>
          <tr><td>Modes</td><td><div className="mode-badges">
        {modeList.map((mode) => (
            <span className="mode-badge" key={mode}>{mode}</span>
        ))}
        </div></td></tr>
        </tbody>
      </table>
    </div>
  );
};

export default SegmentAttributesTable;