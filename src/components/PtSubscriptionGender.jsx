import React, { useEffect, useState } from "react";
import Plot from "react-plotly.js";

const DATASET_COLORS = {
  Microcensus: "#4A90E2",
  Synthetic: "#E07A5F",
};

const VARIABLES = {
    "Male": "0",
    "Female": "1",
};

const PtSubscriptionGender = ({ canton, onClose }) => {
  const [selectedVariable, setSelectedVariable] = useState("0");
  const [data, setData] = useState(null);


  useEffect(() => {
    const selectedCanton = canton || "Zurich"; // Default to "All" if no canton is selected

    fetch("/data/pt_sub_gender.json")
      .then((response) => response.json())
      .then((jsonData) => {
        if (jsonData[selectedCanton]) {
          setData(jsonData[selectedCanton]);
        }

      })
      .catch((error) => console.error("Error loading JSON:", error));
  }, [selectedVariable, canton]);

  if (!data) return <p>Loading...</p>;

  const genders = Object.keys(data["Microcensus"][selectedVariable]);

  return (
    <div className="overlay-panel">
      <h3>{canton || "Zurich"} - Public Transport Subscriptions by Gender</h3> 

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

      {/* Distribution of Pt Subscription by Gender Plot */}
      <Plot
        data={[
          {
            type: "bar",
            x: genders, 
            y:  genders.map((count) =>
              data?.["Microcensus"]?.[selectedVariable]?.[count] ?? 0
            ),
            name: "Microcensus",
            marker: { color: DATASET_COLORS.Microcensus },
            text: genders.map((sub) => (data["Microcensus"][selectedVariable][sub]).toFixed(2)),
            textposition: "auto",
          },
          {
            type: "bar",
            x: genders,
            y:  genders.map((count) =>
              data?.["Synthetic"]?.[selectedVariable]?.[count] ?? 0
            ),
            name: "Synthetic",
            marker: { color: DATASET_COLORS.Synthetic },
            text: genders.map((hour) => (data["Synthetic"][selectedVariable][hour]).toFixed(2)),
            textposition: "auto",
          },
        ]}
        layout={{
            title: { text: "Public Transport Subscriptions by Gender", font: { size: 14 } },
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

export default PtSubscriptionGender