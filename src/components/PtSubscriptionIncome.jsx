import React, { useEffect, useState } from "react";
import Plot from "react-plotly.js";

const DATASET_COLORS = {
  Microcensus: "#4A90E2",
  Synthetic: "#E07A5F",
};

const VARIABLES = {
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
};

const PtSubscriptionIncome = ({ canton, onClose }) => {
  const [selectedVariable, setSelectedVariable] = useState("1");
  const [data, setData] = useState(null);


  useEffect(() => {
    const selectedCanton = canton || "Zurich"; // Default to "All" if no canton is selected

    fetch("/data/pt_sub_income.json")
      .then((response) => response.json())
      .then((jsonData) => {
        if (jsonData[selectedCanton]) {
          setData(jsonData[selectedCanton]);
        }

      })
      .catch((error) => console.error("Error loading JSON:", error));
  }, [selectedVariable, canton]);

  if (!data) return <p>Loading...</p>;

  const incomes = Object.keys(data["Microcensus"][selectedVariable]);

  return (
    <div className="overlay-panel">
      <h3>{canton || "Zurich"} - Public Transport Subscriptions by Income</h3> 

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

      {/* Distribution of Pt Subscription by Income Plot */}
      <Plot
        data={[
          {
            type: "bar",
            x: incomes, 
            y:  incomes.map((count) =>
              data?.["Microcensus"]?.[selectedVariable]?.[count] ?? 0
            ),
            name: "Microcensus",
            marker: { color: DATASET_COLORS.Microcensus },
            text: incomes.map((sub) => (data["Microcensus"][selectedVariable][sub]).toFixed(2)),
            textposition: "auto",
          },
          {
            type: "bar",
            x: incomes,
            y:  incomes.map((count) =>
              data?.["Synthetic"]?.[selectedVariable]?.[count] ?? 0
            ),
            name: "Synthetic",
            marker: { color: DATASET_COLORS.Synthetic },
            text: incomes.map((hour) => (data["Synthetic"][selectedVariable][hour]).toFixed(2)),
            textposition: "auto",
          },
        ]}
        layout={{
            title: { text: "Public Transport Subscriptions by Income", font: { size: 14 } },
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

export default PtSubscriptionIncome