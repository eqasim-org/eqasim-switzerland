import React, { useEffect, useState } from "react";
import Plot from "react-plotly.js";

const DATASET_COLORS = {
  Microcensus: "#4A90E2",
  Synthetic: "#E07A5F",
};

const Histogram = ({ canton, onClose }) => {
  const [euclideanData, setEuclideanData] = useState(null);
  const [networkData, setNetworkData] = useState(null);
  const [selectedMode, setSelectedMode] = useState(null);

  useEffect(() => {
    const selectedCanton = canton || "All";
    fetch(`/data/euclidean_distance_histogram_by_mode.json`)
    .then((response) => response.json())
    .then((jsonData) => {
      if (jsonData[selectedCanton]) {
        setEuclideanData(jsonData[selectedCanton]);
        setSelectedMode(Object.keys(jsonData[selectedCanton])[0]); // Default to first mode
      }
    })
    .catch((error) => console.error("Error loading Euclidean JSON:", error));

  fetch(`/data/network_distance_histogram_by_mode.json`)
    .then((response) => response.json())
    .then((jsonData) => {
      if (jsonData[selectedCanton]) {
        setNetworkData(jsonData[selectedCanton]);
      }
    })
    .catch((error) => console.error("Error loading Network JSON:", error));
}, [canton]);

if (!euclideanData || !networkData) return <p>Loading...</p>;

  const handleModeChange = (event) => {
    setSelectedMode(event.target.value);
  };

  const maxY = Math.max(
    ...euclideanData[selectedMode].microcensus_histogram,
    ...euclideanData[selectedMode].synthetic_histogram,
    ...networkData[selectedMode].microcensus_histogram,
    ...networkData[selectedMode].synthetic_histogram
  );

  const renderPlot = (data, title) => (
    <Plot
      key={title}
      data={[
        {
          type: "bar",
          x: data[selectedMode].bins,
          y: data[selectedMode].microcensus_histogram,
          name: "Microcensus",
          marker: { color: DATASET_COLORS.Microcensus },
          opacity: 0.6,
          hoverinfo: "x+y",
          hovertemplate: "Range: [%{x:.1f} - %{customdata})<br>Percentage: %{y:.2f}%",
          customdata: data[selectedMode].bins.map(bin => (bin + data[selectedMode].bin_width).toFixed(1)) // Compute bin upper bound
        },
        {
          type: "bar",
          x: data[selectedMode].bins,
          y: data[selectedMode].synthetic_histogram,
          name: "Synthetic",
          marker: { color: DATASET_COLORS.Synthetic },
          opacity: 0.6,
          hoverinfo: "x+y",
          hovertemplate: "Range: [%{x:.1f} - %{customdata})<br>Percentage: %{y:.2f}%",
          customdata: data[selectedMode].bins.map(bin => (bin + data[selectedMode].bin_width).toFixed(1)) // Compute bin upper bound
        },
        {
          type: "scatter",
          x: [data[selectedMode].microcensus_mean, data[selectedMode].microcensus_mean],
          y: [0, Math.max(...data[selectedMode].microcensus_histogram) * 0.8],
          mode: "lines",
          line: { color: DATASET_COLORS.Microcensus, dash: "dot" },
          name: "Microcensus Mean",
          hoverinfo: "x",
          hovertemplate: "Mean: %{x:.1f}",
          legendgroup: "microcensus-mean",
        },
        {
          type: "scatter",
          x: [data[selectedMode].microcensus_mean],
          y: [Math.max(...data[selectedMode].microcensus_histogram) * 0.8],
          mode: "text",
          text: [`${data[selectedMode].microcensus_mean.toFixed(1)}`],
          textposition: "top center",
          showlegend: false,
          hoverinfo: "skip",
          legendgroup: "microcensus-mean",
        },
        {
          type: "scatter",
          x: [data[selectedMode].synthetic_mean, data[selectedMode].synthetic_mean],
          y: [0, Math.max(...data[selectedMode].microcensus_histogram) * 0.65],
          mode: "lines",
          line: { color: DATASET_COLORS.Synthetic, dash: "dot" },
          name: "Synthetic Mean",
          hoverinfo: "x",
          hovertemplate: "Mean: %{x:.1f}",
          legendgroup: "synthetic-mean",
        },
        {
          type: "scatter",
          x: [data[selectedMode].synthetic_mean],
          y: [Math.max(...data[selectedMode].microcensus_histogram) * 0.65],
          mode: "text",
          text: [`${data[selectedMode].synthetic_mean.toFixed(1)}`],
          textposition: "top center",
          showlegend: false,
          hoverinfo: "skip",
          legendgroup: "synthetic-mean",
        },
      ]}
      layout={{
        title: { text: `${title}: Mode ${selectedMode}`, font: { size: 14 } },
        xaxis: {
          title: { text: "Distance [m]", font: { size: 12 } },
          tickfont: { size: 10 },
          range: [-data[selectedMode].bin_width, data[selectedMode].bin_width * 25], // Set view extent to first 25 bins
        },
        yaxis: {
          title: { text: "Percentage [%]", font: { size: 12 } },
          tickfont: { size: 10 },
          range: [0, 1.1*maxY],
        },
        margin: { l: 60, r: 20, t: 80, b: 50 },
        height: 300,
        width: 550,
        showlegend: true,
        barmode: "overlay",
        bargap: 0,
        paper_bgcolor: "rgba(255,255,255,0)",
        plot_bgcolor: "rgba(255,255,255,0)",
        annotations: [
          {
            x: 1.45,
            y: 0.1,
            xref: "paper",
            yref: "paper",
            text: `Microcensus: n=${data[selectedMode].microcensus_sample_size}<br>Synthetic: n=${data[selectedMode].synthetic_sample_size}`,
            showarrow: false,
            font: { size: 12 },
            align: "center"
          }
        ],
      }}
    />
  );

  return (
<div className="overlay-panel">
      <h3>{canton || "All"} - Distance Histograms</h3>
      <div>
        {Object.keys(euclideanData).map((mode) => (
          <label key={mode}>
            <input
              type="radio"
              name="mode"
              value={mode}
              checked={selectedMode === mode}
              onChange={handleModeChange}
            />
            {mode}
          </label>
        ))}
      </div>
      <h4>Euclidean Distance</h4>
      {renderPlot(euclideanData, "Euclidean Distance")}
      <h4>Network Distance</h4>
      {renderPlot(networkData, "Network Distance")}
    </div>
  );
};

export default Histogram;
