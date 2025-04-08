import React, { useEffect, useState } from "react";
import Plot from "react-plotly.js";

const MODE_COLORS = {
  car: "#636efa",
  car_passenger: "#ef553b",
  pt: "#00cc96",
  bike: "#ab63fa",
  walk: "#ffa15a",
};

const VARIABLES = {
  "Departure Time": "departure_time",
  "Euclidean Distance": "euclidean_distance",
  "Network Distance": "network_distance",
};

const ModeShareLinePlot = ({ canton }) => {
  const [selectedVariable, setSelectedVariable] = useState("departure_time");
  const [plotData, setPlotData] = useState(null);

  useEffect(() => {
    fetch(`/data/lineplot_${selectedVariable}_data.json`)
      .then((response) => response.json())
      .then((jsonData) => {
        // Select data for the given canton or use "All" as default
        const cantonData = !canton || canton === "All" ? jsonData["All"] : jsonData[canton];

        // Ensure valid data before updating state
        if (cantonData && cantonData.microcensus && cantonData.synthetic) {
          setPlotData(cantonData);
        } else {
          console.error(`No data found for canton: ${canton}`);
          setPlotData(null);
        }
      })
      .catch((error) => console.error(`Error loading ${selectedVariable} data:`, error));
  }, [selectedVariable, canton]);

  if (!plotData) return <p>Loading...</p>;

  const generateTraces = (data, datasetName, lineStyle) => {
    let traces = [];

    // Group data by mode and create traces for each mode
    const uniqueModes = [...new Set(data.map((entry) => entry.mode))];

    uniqueModes.forEach((mode) => {
      const modeData = data.filter((entry) => entry.mode === mode);
      if (modeData.length > 0) {
        traces.push({
          type: "scatter",
          mode: "lines+markers",
          x: modeData.map((entry) => entry.variable_midpoint),
          y: modeData.map((entry) => entry.percentage),
          name: `${mode} (${datasetName})`,
          line: { color: MODE_COLORS[mode] || "#7f7f7f", dash: lineStyle },
        });
      }
    });

    return traces;
  };

  // Ensure tick values and labels are available
  const tickVals = plotData.tick_vals || [];
  const tickLabels = plotData.tick_labels || [];

  // Generate traces for both datasets
  const traces = [
    ...generateTraces(plotData.microcensus, "Microcensus", "solid"),
    ...generateTraces(plotData.synthetic, "Synthetic", "dash"),
  ];

  return (
    <div className="overlay-panel">
      <h3>{canton || "All"} - Mode Share Line Plot</h3>

      {/* Radio buttons to select variable */}
      <div style={{ display: "flex", gap: "15px", marginBottom: "10px" }}>
        {Object.entries(VARIABLES).map(([label, value]) => (
          <label key={value} style={{ display: "flex", alignItems: "center", cursor: "pointer" }}>
            <input
              type="radio"
              value={value}
              checked={selectedVariable === value}
              onChange={() => setSelectedVariable(value)}
              style={{ marginRight: "5px" }}
            />
            {label}
          </label>
        ))}
      </div>

      {/* Plotly Chart */}
      <Plot
        data={traces}
        layout={{
          title: `Mode Share Percentage vs ${selectedVariable.replace("_", " ").toUpperCase()} - ${canton || "All"}`,
          xaxis: {
            title: {
                text: `${selectedVariable.replace("_", " ").replace(/\b\w/g, (char) => char.toUpperCase())}`,
                font: { size: 14 },
                standoff: 20},
            tickmode: "array",
            tickvals: tickVals,
            ticktext: tickLabels,
            tickangle: 45,
          },
          yaxis: { title: { text: "Mode Share [%]", font: { size: 14 } }, },
          legend: { title: { text: "Modes" } },
          width: 880,
          height: 450,
          paper_bgcolor: "rgba(255,255,255,0)",
          plot_bgcolor: "rgba(255,255,255,0)",
          margin: { l: 60, r: 20, t: 20, b: 80 },
        }}
      />
    </div>
  );
};

export default ModeShareLinePlot;
