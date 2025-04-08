import React, { useEffect, useState } from "react";
import Plot from "react-plotly.js";

const DATASET_COLORS = {
  Microcensus: "#4A90E2",
  Synthetic: "#E07A5F",
};

const AverageDist = ({ canton, onClose }) => {
  const [data, setData] = useState(null);

  useEffect(() => {
    const selectedCanton = canton || "All"; // Default to "All" if no canton is selected

    fetch("/data/avg_dist_data.json")
      .then((response) => response.json())
      .then((jsonData) => {
        if (jsonData[selectedCanton]) {
          setData(jsonData[selectedCanton]); // Load data for selected canton or "All"
        }
      })
      .catch((error) => console.error("Error loading JSON:", error));
  }, [canton]);

  if (!data) return <p>Loading...</p>;

  const modes = Object.keys(data["Microcensus"]); // Get modes

  const total_sample_microcensus = Math.round(
    Object.values(data["Microcensus"]).reduce((sum, mode) => sum + mode.sample_size, 0)
  ).toLocaleString();
  
  const total_sample_synthetic = Math.round(
    Object.values(data["Synthetic"]).reduce((sum, mode) => sum + mode.sample_size, 0)
  ).toLocaleString();
  

  return (
    <div className="overlay-panel">
      <h3>{canton || "All"} - Average Distance by Mode</h3> 

      <p><b>Sample Sizes:</b> Microcensus: {total_sample_microcensus}, Synthetic: {total_sample_synthetic}</p>

      {/* Euclidean Distance Plot */}
      <Plot
        data={[
          {
            type: "bar",
            x: modes,
            y: modes.map((mode) => (data["Microcensus"][mode].euclidean_distance / 1000).toFixed(1)),
            name: "Microcensus",
            marker: { color: DATASET_COLORS.Microcensus },
            text: modes.map((mode) => (data["Microcensus"][mode].euclidean_distance / 1000).toFixed(1)),
            textposition: "auto",
          },
          {
            type: "bar",
            x: modes,
            y: modes.map((mode) => (data["Synthetic"][mode].euclidean_distance / 1000).toFixed(1)),
            name: "Synthetic",
            marker: { color: DATASET_COLORS.Synthetic },
            text: modes.map((mode) => (data["Synthetic"][mode].euclidean_distance / 1000).toFixed(1)),
            textposition: "auto",
          },
        ]}
        layout={{
            title: { text: "Euclidean Distance", font: { size: 14 } },
            xaxis: { 
                title: { text: "Mode", font: { size: 12 } }, 
                tickangle: -45, 
                tickfont: { size: 10 } 
            },
            yaxis: { 
                title: { text: "Average Distance [km]", font: { size: 12 } }, 
                tickfont: { size: 10 } 
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

      {/* Network Distance Plot */}
      <Plot
        data={[
          {
            type: "bar",
            x: modes,
            y: modes.map((mode) => (data["Microcensus"][mode].network_distance / 1000).toFixed(1)),
            name: "Microcensus",
            marker: { color: DATASET_COLORS.Microcensus },
            text: modes.map((mode) => (data["Microcensus"][mode].network_distance / 1000).toFixed(1)),
            textposition: "auto",
          },
          {
            type: "bar",
            x: modes,
            y: modes.map((mode) => (data["Synthetic"][mode].network_distance / 1000).toFixed(1)),
            name: "Synthetic",
            marker: { color: DATASET_COLORS.Synthetic },
            text: modes.map((mode) => (data["Synthetic"][mode].network_distance / 1000).toFixed(1)),
            textposition: "auto",
          },
        ]}
        layout={{
            title: { text: "Network Distance", font: { size: 14 } },
            xaxis: { 
                title: { text: "Mode", font: { size: 12 } }, 
                tickangle: -45, 
                tickfont: { size: 10 } 
            },
            yaxis: { 
                title: { text: "Average Distance [km]", font: { size: 12 } }, 
                tickfont: { size: 10 } 
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

export default AverageDist;