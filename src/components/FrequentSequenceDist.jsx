import React, { useEffect, useState } from "react";
import Plot from "react-plotly.js";

const DATASET_COLORS = {
  Microcensus: "#4A90E2",
  Synthetic: "#E07A5F",
};

const FrequentSequence = ({ canton, onClose }) => {
  const [data, setData] = useState(null);

  useEffect(() => {
    const selectedCanton = canton || "Zurich";

    fetch("/data/frequent_sequences.json")
      .then((response) => response.json())
      .then((jsonData) => {
        if (jsonData[selectedCanton]) {
          setData(jsonData[selectedCanton]);
        }

      })
      .catch((error) => console.error("Error loading JSON:", error));
  }, [canton]);

  if (!data) return <p>Loading...</p>;

  const act_sequence = Object.keys(data["Microcensus"]);
  

  return (
    <div className="overlay-panel">
      <h3>{canton || "Zurich"} - Frequent Activity Sequences</h3> 

      {/* <p><b>Sample Sizes:</b> Microcensus: {total_sample_microcensus}, Synthetic: {total_sample_synthetic}</p> */}

      {/* Euclidean Distance Plot */}
      <Plot
        data={[
          {
            type: "bar",
            x: act_sequence,
            y: act_sequence.map((sequence) => (data["Microcensus"][sequence])),
            name: "Microcensus",
            marker: { color: DATASET_COLORS.Microcensus },
            text: act_sequence.map((sequence) => (data["Microcensus"][sequence]).toFixed(2)),
            textposition: "auto",
          },
          {
            type: "bar",
            x: act_sequence,
            y: act_sequence.map((sequence) => (data["Synthetic"][sequence])),
            name: "Synthetic",
            marker: { color: DATASET_COLORS.Synthetic },
            text: act_sequence.map((sequence) => (data["Synthetic"][sequence]).toFixed(2)),
            textposition: "auto",
          },
        ]}
        layout={{
            title: { text: "Frequent Activity Sequences", font: { size: 14 } },
            xaxis: { 
                title: { text: "Activity Sequence", font: { size: 12 } }, 
                tickangle: -45, 
                tickfont: { size: 10 } 
            },
            yaxis: { 
                title: { text: "Proportion", font: { size: 12 } }, 
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

export default FrequentSequence;