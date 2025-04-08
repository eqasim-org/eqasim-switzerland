import React, { useEffect, useState } from "react";
import Plot from "react-plotly.js";

// Define mode colors
const MODE_COLORS = {
  car: "#636efa",
  car_passenger: "#ef553b",
  pt: "#00cc96",
  bike: "#ab63fa",
  walk: "#ffa15a",
};

// Define distance bins (order matters)
const DISTANCE_CATEGORIES = ["0-1000", "1000-5000", "5000-25000", "25000+"];

// Define datasets
const DATASETS = ["Microcensus", "Synthetic"];

const StackedBarPlots = ({ canton }) => {
  const [euclideanData, setEuclideanData] = useState(null);
  const [networkData, setNetworkData] = useState(null);

  useEffect(() => {
    fetch(`/data/euclidean_distance_mode_share_data.json`)
      .then((response) => response.json())
      .then((jsonData) => {
        const cantonData =
          !canton || canton === "All"
            ? jsonData.all
            : jsonData.cantons[canton];
        setEuclideanData(cantonData);
      })
      .catch((error) => console.error("Error loading Euclidean Data:", error));

    fetch(`/data/network_distance_mode_share_data.json`)
      .then((response) => response.json())
      .then((jsonData) => {
        const cantonData =
          !canton || canton === "All"
            ? jsonData.all
            : jsonData.cantons[canton];
        setNetworkData(cantonData);
      })
      .catch((error) => console.error("Error loading Network Data:", error));
  }, [canton]);

  if (!euclideanData || !networkData) return <p>Loading...</p>;

  // Function to generate traces with axis index offset
  const generateTraces = (data, datasetName, rowOffset = 1) => {
    let traces = [];

    DISTANCE_CATEGORIES.forEach((category, i) => {
      DATASETS.forEach((dataset) => {
        const datasetData = data.filter(
          (entry) =>
            entry.distance_category === category && entry.dataset === dataset
        );

        datasetData.forEach((entry) => {
          const axisIndex = i + 1 + (rowOffset - 1) * 4;
          traces.push({
            type: "bar",
            x: [dataset],
            y: [entry.percentage],
            name: entry.mode,
            text: [entry.percentage >= 5 ? `${entry.percentage}%` : ""],
            hovertemplate: `Dataset: ${datasetName}<br>Mode: ${entry.mode}<br>Percentage: %{y:.1f}%`,
            marker: { color: MODE_COLORS[entry.mode] },
            opacity: 0.7,
            legendgroup: entry.mode,
            showlegend: dataset === "Microcensus" && rowOffset === 1 && i === 0,
            xaxis: `x${axisIndex}`,
            yaxis: `y${axisIndex}`,
            textfont: { size: 10 },
          });
        });
      });

      // Add category label as scatter
      const labelAxis = i + 1 + (rowOffset - 1) * 4;
      traces.push({
        type: "scatter",
        mode: "text",
        x: ["Microcensus"],
        y: [105],
        text: `                         ${category} m`,
        textposition: "top center",
        showlegend: false,
        xaxis: `x${labelAxis}`,
        yaxis: `y${labelAxis}`,
      });
    });

    return traces;
  };

  const allTraces = [
    ...generateTraces(euclideanData, "Euclidean", 1),
    ...generateTraces(networkData, "Network", 2),
  ];

  const annotations = [
    {
      text: "Euclidean Distance",
      x: 0.5,
      y: 1.07,
      xref: "paper",
      yref: "paper",
      showarrow: false,
      font: { size: 17, color: "#000" },
      xanchor: "center",
    },
    {
      text: "Network Distance",
      x: 0.5,
      y: 0.46,
      xref: "paper",
      yref: "paper",
      showarrow: false,
      font: { size: 17, color: "#000" },
      xanchor: "center",
    },
  ];

  return (
    <div className="overlay-panel">
      <h3>{canton || "All"} - Mode Share by Distance Category</h3>

      <Plot
        data={allTraces}
        layout={{
          title: "",
          grid: { rows: 2, columns: 4, pattern: "independent" },
          barmode: "stack",
          bargap: 0.15,
          width: 800,
          height: 800,
          paper_bgcolor: "rgba(255,255,255,0)",
          plot_bgcolor: "rgba(255,255,255,0)",
          showlegend: true,
          legend: {
            title: { text: "Modes" },
            orientation: "h",
            y: -0.05,
            x: 0.5,
            xanchor: "center",
          },
          margin: { l: 60, r: 20, t: 70, b: 0 },
          annotations,
          ...Object.fromEntries(
            DISTANCE_CATEGORIES.flatMap((_, i) => [
              [`xaxis${i + 1}`, { tickfont: { size: 10 }, showgrid: false }],
              [
                `yaxis${i + 1}`,
                {
                  title: i === 0 ? "Euclidean" : "",
                  tickfont: { size: 10 },
                  range: [0, 110],
                  showgrid: true,
                },
              ],
              [`xaxis${i + 5}`, { tickfont: { size: 10 }, showgrid: false }],
              [
                `yaxis${i + 5}`,
                {
                  title: i === 0 ? "Network" : "",
                  tickfont: { size: 10 },
                  range: [0, 110],
                  showgrid: true,
                },
              ],
            ])
          ),
        }}
      />
    </div>
  );
};

export default StackedBarPlots;
