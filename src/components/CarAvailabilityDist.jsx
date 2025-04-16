import React, { useEffect, useState } from "react";
import Plot from "react-plotly.js";

const DATASET_COLORS = {
  Microcensus: "#4A90E2",
  Synthetic: "#E07A5F",
};

const CarAvailability = ({ canton, onClose }) => {
  const [data, setData] = useState(null);

  useEffect(() => {
    const selectedCanton = canton || "Zurich"; // Default to "All" if no canton is selected

    fetch("/data/car_availability.json")
      .then((response) => response.json())
      .then((jsonData) => {
        if (jsonData[selectedCanton]) {
          setData(jsonData[selectedCanton]); // Load data for selected canton or "All"
        }

    })
      .catch((error) => console.error("Error loading JSON:", error));
  }, [canton]);

  if (!data) return <p>Loading...</p>;

  const car_class = Object.keys(data["Microcensus"]);
  

  return (
    <div className="overlay-panel">
      <h3>{canton || "Zurich"} - Car Availability Class</h3> 

      {/* <p><b>Sample Sizes:</b> Microcensus: {total_sample_microcensus}, Synthetic: {total_sample_synthetic}</p> */}

      {/* Euclidean Distance Plot */}
      <Plot
        data={[
          {
            type: "bar",
            x: car_class,
            y: car_class.map((car) => (data["Microcensus"][car])),
            name: "Microcensus",
            marker: { color: DATASET_COLORS.Microcensus },
            text: car_class.map((car) => (data["Microcensus"][car]).toFixed(2)),
            textposition: "auto",
          },
          {
            type: "bar",
            x: car_class,
            y: car_class.map((car) => (data["Synthetic"][car])),
            name: "Synthetic",
            marker: { color: DATASET_COLORS.Synthetic },
            text: car_class.map((car) => (data["Synthetic"][car]).toFixed(2)),
            textposition: "auto",
          },
        ]}
        layout={{
            title: { text: "Car Availability Class", font: { size: 14 } },
            xaxis: { 
                title: { text: "Car Class", font: { size: 12 } }, 
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

export default CarAvailability;