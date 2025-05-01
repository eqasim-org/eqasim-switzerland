import React, { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import PtSubscriptionAge from "./PtSubscriptionAge";
import PtSubscriptionGender from "./PtSubscriptionGender";
import PtSubscriptionIncome from "./PtSubscriptionIncome";

const DATASET_COLORS = {
  Microcensus: "#4A90E2",
  Synthetic: "#E07A5F",
};

const PtSubscription = ({ canton, onClose }) => {
  const [data, setData] = useState(null);

  useEffect(() => {
    const selectedCanton = canton || "Zurich"; // Default to "All" if no canton is selected

    fetch("/data/pt_subscriptions.json")
      .then((response) => response.json())
      .then((jsonData) => {
        if (jsonData[selectedCanton]) {
          setData(jsonData[selectedCanton]); // Load data for selected canton or "All"
        }

    })
      .catch((error) => console.error("Error loading JSON:", error));
  }, [canton]);

  if (!data) return <p>Loading...</p>;

  const pt_subscription = Object.keys(data["Microcensus"]);
  

  return (
    <div className="overlay-panel">
      <h3>{canton || "Zurich"} - Public Transport Subscription Types</h3> 

      {/* <p><b>Sample Sizes:</b> Microcensus: {total_sample_microcensus}, Synthetic: {total_sample_synthetic}</p> */}

      <Plot
        data={[
          {
            type: "bar",
            x: pt_subscription,
            y: pt_subscription.map((sub) => (data["Microcensus"][sub])),
            name: "Microcensus",
            marker: { color: DATASET_COLORS.Microcensus },
            text: pt_subscription.map((sub) => (data["Microcensus"][sub]).toFixed(2)),
            textposition: "auto",
          },
          {
            type: "bar",
            x: pt_subscription,
            y: pt_subscription.map((sub) => (data["Synthetic"][sub])),
            name: "Synthetic",
            marker: { color: DATASET_COLORS.Synthetic },
            text: pt_subscription.map((sub) => (data["Synthetic"][sub]).toFixed(2)),
            textposition: "auto",
          },
        ]}
        layout={{
            title: { text: "Public Transport Subscriptions", font: { size: 14 } },
            xaxis: { 
                title: { text: "Subscription Type", font: { size: 12 } }, 
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
      <PtSubscriptionAge canton={canton} onClose={onClose} ></PtSubscriptionAge>
      <PtSubscriptionGender canton={canton} onClose={onClose}></PtSubscriptionGender>
      <PtSubscriptionIncome canton={canton} onClose={onClose}></PtSubscriptionIncome>
    </div>
  );
};

export default PtSubscription;