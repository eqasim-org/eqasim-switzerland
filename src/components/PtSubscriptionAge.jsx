import React, { useEffect, useState } from "react";
import Plot from "react-plotly.js";

const DATASET_COLORS = {
  Microcensus: "#4A90E2",
  Synthetic: "#E07A5F",
};

const VARIABLES = {
    "[6, 15)": "[6, 15)",
    "[15, 18)": "[15, 18)",
    "[18, 24)": "[18, 24)",
    "[24, 30)": "[24, 30)",
    "[30, 45)": "[30, 45)",
    "[45, 65)": "[45, 65)",
    "[65, 80)": "[65, 80)",
};

const PtSubscriptionAge = ({ canton, onClose }) => {
  const [selectedVariable, setSelectedVariable] = useState("[6, 15)");
  const [data, setData] = useState(null);


  useEffect(() => {
    const selectedCanton = canton || "Zurich"; // Default to "All" if no canton is selected

    fetch("/data/pt_sub_age.json")
      .then((response) => response.json())
      .then((jsonData) => {
        if (jsonData[selectedCanton]) {
          setData(jsonData[selectedCanton]);
        }

      })
      .catch((error) => console.error("Error loading JSON:", error));
  }, [selectedVariable, canton]);

  if (!data) return <p>Loading...</p>;

  const ages = Object.keys(data["Microcensus"][selectedVariable]);

  return (
    <div className="overlay-panel">
      <h3>{canton || "Zurich"} - Public Transport Subscriptions by Age</h3> 

      {/* Radio buttons to select variable */}
      <div style={{ display: "flex", gap: "15px", marginBottom: "10px" }}>
        {Object.entries(VARIABLES).map(([label, value]) => (
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

      {/* Distribution of Pt Subscription by Age Plot */}
      <Plot
        data={[
          {
            type: "bar",
            x: ages, 
            y:  ages.map((count) =>
              data?.["Microcensus"]?.[selectedVariable]?.[count] ?? 0
            ),
            name: "Microcensus",
            marker: { color: DATASET_COLORS.Microcensus },
            text: ages.map((sub) => (data["Microcensus"][selectedVariable][sub]).toFixed(2)),
            textposition: "auto",
          },
          {
            type: "bar",
            x: ages,
            y:  ages.map((count) =>
              data?.["Synthetic"]?.[selectedVariable]?.[count] ?? 0
            ),
            name: "Synthetic",
            marker: { color: DATASET_COLORS.Synthetic },
            text: ages.map((hour) => (data["Synthetic"][selectedVariable][hour]).toFixed(2)),
            textposition: "auto",
          },
        ]}
        layout={{
            title: { text: "Public Transport Subscriptions by Age", font: { size: 14 } },
            xaxis: { 
                title: { text: "Public Transport Subscription Type", font: { size: 12 } }, 
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

export default PtSubscriptionAge