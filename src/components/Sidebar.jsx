import React, { useState, useEffect  } from "react";
import "./Sidebar.css";

// For mode filter (only show modes available in each canton)
import modesByCanton from "../utils/modes_by_canton.json";

// Sidebar Modules / Graphs
import AverageDist from "./AverageDist";
import Histogram from "./Histogram";
import StackedBarPlot from "./StackedBarPlot";
import ChoroplethControls from "./ChoroplethControls";
import CantonModeShareTable from "./CantonModeShareTable"; 
import ModeShareLinePlot from "./ModeShareLinePlot";
import SegmentAttributesTable from "./SegmentAttributesTable";
import SegmentVolumeHistogram from "./SegmentVolumeHistogram";

const Sidebar = ({canton, isOpen, toggleSidebar, onExpandGraph, setCanton, resetMapView, updateMapSymbology,
  selectedNetworkModes, setSelectedNetworkModes, selectedNetworkFeature, setVisualizeLinkId}) => {
    
    // ======================= INITIALIZE VARIABLES =======================
    
    const [selectedGraph, setSelectedGraph] = useState(null); // Current module
    const [selectedMode, setSelectedMode] = useState("None"); // Choropleth mode
    const [selectedDataset, setSelectedDataset] = useState("Microcensus"); // Choropleth dataset
    const [availableModes, setAvailableModes] = useState([]); // Available modes for network filter
    
    // ======================= GENERAL FEATURES (BUTTONS / DROPDOWN) =======================
    // Push current module to Map
    const handleGraphSelection = (event) => {
      const graph = event.target.value;
      setSelectedGraph(graph);
      onExpandGraph(graph);
      
      // Set corresponding default selected modes per module
      if (graph === "Volumes") {
        setSelectedNetworkModes(["car"]);
      }
      
      if (graph === "Network") {
        setSelectedNetworkModes(["all"]);
      }
      
      if (graph != "Network" && graph != "Volumes") {
        setSelectedNetworkModes(["all"]);
      }
    };
    
    // Handle home button click
    const handleHome = () => {
      setSelectedGraph(null);
      onExpandGraph(null);
    };
    
    // Handle reset button click
    const handleReset = () => {
      setCanton(null);
      setSelectedMode("None");
      setSelectedNetworkModes(["all"]);
      updateMapSymbology("None", selectedDataset);
      resetMapView();
    };


    // ======================= MATSIM NETWORK MODULE =======================
    
    // Get available modes per canton
    useEffect(() => {
      if (canton && modesByCanton[canton]) {
        setAvailableModes(
          modesByCanton[canton].filter(mode =>
            !["car_passenger", "truck", "train", "other", "pt"].includes(mode)
          )
        );
      } else {
        setAvailableModes([]);
      }
    }, [canton]);
    
    // Push to Map the selected modes
    const handleModeChange = (event) => {
      const selectedOptions = Array.from(event.target.selectedOptions).map((option) => option.value);
      
      if (selectedOptions.includes("all") || selectedOptions.length === 0) {
        setSelectedNetworkModes(["all"]);
      } else {
        setSelectedNetworkModes(selectedOptions);
      }
    };  
    
    return (
      <div className={`floating-panel ${isOpen ?  // Sets the css for sidebar width
        (selectedGraph === "Graph 3" || selectedGraph === "Graph 4" ? "expanded-graph3" : 
          selectedGraph === "Choropleth"  || selectedGraph === "Network" ? "open" : 
          selectedGraph ? "expanded" : "open") 
          : "collapsed"}`}>
          <button className="toggle-button" onClick={toggleSidebar}>{isOpen ? "✕" : "☰"}</button>          
          
          {isOpen && (
            <div className="floating-content">
            <br />
            
            {/* Home, Reset, and Graph Selection */}
            <div className="button-row">
            <div className="button-group">
            <button className={`home-button ${!selectedGraph ? "active" : ""}`} onClick={handleHome}>
            Home
            </button>
            <button className="reset-button" onClick={handleReset}>
            Reset
            </button>
            <select className="graph-dropdown" value={selectedGraph || ""} onChange={handleGraphSelection}>
            <option value="">Select a Graph</option>
            <option value="Choropleth">Choropleth</option>
            <option value="Network">MATSim Network</option>
            <option value="Volumes">Simulation Volumes</option>
            <option value="Graph 1">Average Distance</option>
            <option value="Graph 2">Histogram</option>
            <option value="Graph 3">Stacked Bar Plot</option>
            <option value="Graph 4">Line Plot</option>
            </select>
            </div>
            </div>
            
            {/* Default View */}
            {!selectedGraph && (
              <div className="home-message">
              <p>Select a canton and a visualization to get started!</p>
              </div>
            )}
            
            {/* Rendering for graphs */}
            {selectedGraph === "Graph 1" && <div className="plot-container"><AverageDist canton={canton || "All"} /></div>}
            {selectedGraph === "Graph 2" && <div className="plot-container"><Histogram canton={canton || "All"} /></div>}
            {selectedGraph === "Graph 3" && <div className="plot-container"><StackedBarPlot canton={canton || "All"} /></div>}
            {selectedGraph === "Graph 4" && <div className="plot-container"><ModeShareLinePlot canton={canton || "All"} /></div>}
            
            {/* Mode Share Choropleth Selection */}
            {selectedGraph === "Choropleth" && (
              <div>
              <ChoroplethControls
              selectedMode={selectedMode}
              setSelectedMode={setSelectedMode}
              selectedDataset={selectedDataset}
              setSelectedDataset={setSelectedDataset}
              updateMapSymbology={updateMapSymbology}
              />
              <CantonModeShareTable canton={canton} selectedDataset={selectedDataset} selectedMode={selectedMode} />
              </div>
            )}
            
            {/* Network Module */}
            {selectedGraph === "Network" && (
              
              <div>
              <div className="mode-filter-container">
              <label className="mode-filter-label">Filter by Mode:</label>
              <select
              multiple
              value={selectedNetworkModes}
              onChange={handleModeChange}
              className="mode-filter-select"
              >
              <option value="all">All</option>
              {availableModes.map((mode) => (
                <option key={mode} value={mode}>
                {mode.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())}
                </option>
              ))}
              </select>
              </div>  
              {selectedNetworkFeature && (
                <SegmentAttributesTable properties={selectedNetworkFeature[0]} />
              )}
              </div>
            )}
            
            {selectedGraph === "Volumes" && (
              <div className="plot-container">
              {selectedNetworkFeature && (
                <SegmentAttributesTable properties={selectedNetworkFeature[0]} />
              )}
              
              {selectedNetworkFeature ? (
                <SegmentVolumeHistogram
                linkId={selectedNetworkFeature.map(f => f.id)}
                setVisualizeLinkId={setVisualizeLinkId}
                canton={canton}
                />
              ) : (
                <p style={{ padding: "1rem", fontStyle: "italic", color: "#555" }}>
                Click a canton and/or segment to see hourly volumes.
                </p>
              )}
              </div>
            )}
            
            </div>
          )}
          </div>
        );
      };
      
      export default Sidebar;
      