import React, { useEffect, useState } from "react";
import Plot from "react-plotly.js";

const GenericBarPlot = ({
    dataFile = "/data/pt_sub_age.json",
    title = "Public Transport Subscriptions by Age",
    xAxisTitle = "Public Transport Subscription Type",
    yAxisTitle = "Proportion",
    dataKeys = ["Microcensus", "Synthetic"],
    datasetColors = {
    Microcensus: "#4A90E2",
    Synthetic: "#E07A5F",
    },
    variables,
    defaultVariable,
    canton = "Zurich",
    onClose,
}) => {
    const [selectedVariable, setSelectedVariable] = useState(defaultVariable);
    const [data, setData] = useState(null);

    useEffect(() => {
    fetch(dataFile)
        .then((res) => res.json())
        .then((json) => {
        if (json[canton]) {
            setData(json[canton]);
        }
        })
        .catch((err) => console.error("Error loading JSON:", err));
    }, [selectedVariable, canton, dataFile]);

    if (!data) return <p>Loading...</p>;

    const xLabels = Object.keys(data[dataKeys[0]][selectedVariable]);

    return (
    <div className="overlay-panel">
      <h3>{canton} - {title}</h3>

      <div style={{ display: "flex", gap: "15px", marginBottom: "10px" }}>
        {Object.entries(variables).map(([label, value]) => (
          <label key={value} style={{ display: "flex", alignItems: "center", cursor: "pointer", whiteSpace: "nowrap" }}>
            <input
              type="radio"
              value={value}
              checked={selectedVariable === value}
              onChange={() => setSelectedVariable(value)}
              style={{ marginRight: "5px" }}
            />
            <span>{label}</span>
          </label>
        ))}
      </div>

      <Plot
        data={dataKeys.map((key) => ({
          type: "bar",
          x: xLabels,
          y: xLabels.map((label) => data?.[key]?.[selectedVariable]?.[label] ?? 0),
          name: key,
          marker: { color: datasetColors[key] || "#ccc" },
          text: xLabels.map((label) => data[key][selectedVariable][label]?.toFixed(2)),
          textposition: "auto",
        }))}
        layout={{
          title: { text: title, font: { size: 14 } },
          xaxis: {
            title: { text: xAxisTitle, font: { size: 12 } },
            tickangle: -45,
            tickfont: { size: 10 },
          },
          yaxis: {
            title: { text: yAxisTitle, font: { size: 12 } },
            tickfont: { size: 10 },
          },
          margin: { l: 50, r: 20, t: 120, b: 80 },
          height: 350,
          width: 550,
          showlegend: true,
          barmode: "group",
          paper_bgcolor: "rgba(255,255,255,0)",
          plot_bgcolor: "rgba(255,255,255,0)",
        }}
      />
    </div>
  );
};

export default GenericBarPlot;
