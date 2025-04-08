import React, { useEffect, useState } from "react";
import Plot from "react-plotly.js";

const SegmentVolumeHistogram = ({ linkId, setVisualizeLinkId, canton}) => {
  const [volumeData, setVolumeData] = useState(null);

  // Normalize to always handle as array
  const linkIds = Array.isArray(linkId) ? linkId : [linkId];

  useEffect(() => {
    if (!linkIds || linkIds.length === 0) return;

    fetch(`/data/matsim/${canton}_link_traffic_volumes.json`)
      .then((res) => res.json())
      .then((data) => {
        const filtered = data.filter((entry) =>
          linkIds.includes(entry.link_id.toString())
        );
        const mapped = Object.fromEntries(
          filtered.map((entry) => [entry.link_id.toString(), entry.hourly_avg_volumes])
        );
        setVolumeData(mapped);
      })
      .catch((err) => console.error("Error loading volume data:", err));
  }, [linkId, canton]);

  if (!volumeData) return <p>Loading volume data...</p>;

  return (
    <div className="plot-container">
      {linkIds.map((id) => {
        const hourlyData = volumeData[id.toString()];
        if (!hourlyData) return null;

        const hours = Object.keys(hourlyData);
        const formattedLabels = hours.map((h) => {
          const match = h.match(/HRS(\d+)-(\d+)avg/);
          if (match) {
            const hour = parseInt(match[1], 10);
            return hour.toString().padStart(2, "0") + ":00";
          }
          return h;
        });

        const tickvals = formattedLabels.filter((_, i) => i % 2 === 0);
        const values = hours.map((h) => hourlyData[h]);

        return (
          <div key={id}>
            <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
              <h4 style={{ margin: 0 }}>Hourly Volume for Link {id}</h4>
              <button className="graph-button small"
                onClick={() => setVisualizeLinkId(id)}
                >Visualize</button>
            </div>
            <Plot
              data={[
                {
                  x: formattedLabels,
                  y: values,
                  type: "bar",
                  marker: { color: "#17becf" },
                },
              ]}
              layout={{
                margin: { t: 30, r: 10, l: 40, b: 100 },
                xaxis: {
                  title: { text: "Hour", standoff: 20 },
                  tickangle: -45,
                  tickvals,
                  automargin: true,
                },
                yaxis: { title: "Avg Vehicles/hour" },
                height: 300,
                width: 525,
                paper_bgcolor: "rgba(255,255,255,0)",
                plot_bgcolor: "rgba(255,255,255,0)",
              }}
            />
          </div>
        );
      })}
    </div>
  );
};

export default SegmentVolumeHistogram;
