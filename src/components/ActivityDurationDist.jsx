import React, { useEffect, useState } from "react";
import Plot from "react-plotly.js";

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

const ActivityDurationDist = ({ canton, onClose }) => {
  const [selectedVariable, setSelectedVariable] = useState("work");
  const [data, setData] = useState(null);


  useEffect(() => {
    const selectedCanton = canton || "Zurich"; // Default to "All" if no canton is selected

    fetch("/data/activity_durations.json")
      .then((response) => response.json())
      .then((jsonData) => {
        if (jsonData[selectedCanton]) {
          setData(jsonData[selectedCanton]); // Load data for selected canton or "All"
        }

      })
      .catch((error) => console.error("Error loading JSON:", error));
  }, [selectedVariable, canton]);

  if (!data) return <p>Loading...</p>;

  const durations = Object.keys(data["Microcensus"][selectedVariable]);

  return (
    <div className="overlay-panel">
      <h3>{canton || "Zurich"} - Activity Durations</h3> 

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

      {/* Distribution of Durations Plot */}
      <Plot
        data={[
          {
            type: "bar",
            x: durations, 
            y:  durations.map((count) =>
              data?.["Microcensus"]?.[selectedVariable]?.[count] ?? 0
            ),
            name: "Microcensus",
            marker: { color: DATASET_COLORS.Microcensus },
            text: durations.map((count) => (data["Microcensus"][selectedVariable][count])),
            textposition: "auto",
          },
          {
            type: "bar",
            x: durations,
            y:  durations.map((count) =>
              data?.["Synthetic"]?.[selectedVariable]?.[count] ?? 0
            ),
            name: "Synthetic",
            marker: { color: DATASET_COLORS.Synthetic },
            text: durations.map((count) => (data["Synthetic"][selectedVariable][count])),
            textposition: "auto",
          },
        ]}
        layout={{
            title: { text: "Activity Duration Distribution", font: { size: 14 } },
            xaxis: { 
                title: { text: "Duration [HH:mm:ss]", font: { size: 12 } }, 
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

export default ActivityDurationDist;