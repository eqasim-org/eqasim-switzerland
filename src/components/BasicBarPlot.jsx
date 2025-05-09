import React, { useEffect, useState } from "react";
import Plot from "react-plotly.js";

const BasicBarPlot = ({
    dataFile = "/data/frequent_sequences.json",
    title = "Frequent Activity Sequences",
    xAxisTitle = "Activity Sequence",
    yAxisTitle = "Proportion",
    dataKeys = ["Microcensus", "Synthetic"],
    datasetColors = {
    Microcensus: "#4A90E2",
    Synthetic: "#E07A5F",
    },
    canton = "Zurich",
    onClose,
}) => {
    const [data, setData] = useState(null);

    useEffect(() => {
    fetch(dataFile)
        .then((response) => response.json())
        .then((json) => {
        if (json[canton]) {
            setData(json[canton]);
        }
        })
        .catch((error) => console.error("Error loading JSON:", error));
    }, [canton, dataFile]);

    if (!data) return <p>Loading...</p>;

    const xLabels = Object.keys(data[dataKeys[0]]);

    return (
    <div className="overlay-panel">
        <h3>{canton} - {title}</h3>

        <Plot
        data={dataKeys.map((key) => ({
            type: "bar",
            x: xLabels,
            y: xLabels.map((label) => data?.[key]?.[label] ?? 0),
            name: key,
            marker: { color: datasetColors[key] || "#ccc" },
            text: xLabels.map((label) => {
            const val = data?.[key]?.[label];
            return typeof val === "number" ? val.toFixed(2) : "";
            }),
            textposition: "auto",
        }))}
        layout={{
            title: { text: title, font: { size: 14 } },
            xaxis: {
            title: { text: xAxisTitle, font: { size: 12 } },
            tickangle: -45,
            tickfont: { size: 10 },
            },
            yaxis: {
            title: { text: yAxisTitle, font: { size: 12 } },
            tickfont: { size: 10 },
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

export default BasicBarPlot;
