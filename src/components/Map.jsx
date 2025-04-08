import { useRef, useEffect, useState } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import bboxCache from '../utils/bboxCanton.json'; 
import "./Loading.css" // loading screen for network

const Map = ({ mapRef, setClickedCanton, isSidebarOpen, isGraphExpanded, searchCanton, selectedMode,
  selectedDataset, selectedNetworkModes, setSelectedNetworkFeature, visualizeLinkId }) => {
    
    // ======================= INITIALIZE VARIABLES =======================
    
    // Initialize map reference
    const mapContainerRef = useRef(null);
    
    // Indicators for sidebar size
    const graphExpandedRef = useRef(isGraphExpanded);
    
    // Set modeshare data values
    const [modeShareData, setModeShareData] = useState(null);
    
    // Set max pct of each mode (for choropleth normalization)
    const [maxSharePerMode, setMaxSharePerMode] = useState(null);
    
    // Add map loading when loading network
    const [isLoadingNetwork, setIsLoadingNetwork] = useState(false);

    // Keep track of select network modes
    const selectedNetworkModesRef = useRef(selectedNetworkModes);
    
    
    // ======================= INITIALIZE MAP AND HANDLE CANTON SELECTION =======================
    
    useEffect(() => {
      
      mapboxgl.accessToken = 'pk.eyJ1IjoiYW5kd29vIiwiYSI6ImNrMjlnYnNkdTEwMHozaG5wamJvZHJyangifQ.6M4eeri_Ubmo7NedQT7NuQ';
      
      const map = new mapboxgl.Map({
        container: mapContainerRef.current,
        style: 'mapbox://styles/mapbox/light-v10',
        center: [8.1642, 46.7592],
        zoom: 7,
      });
      
      mapRef.current = map;
      
      // Fetch Swiss cantons from geojson
      fetch('/data/TLM_KANTONSGEBIET.geojson')
      .then(response => response.json())
      .then(cantonGeoJSON => {
        
        map.addSource('cantons', { type: 'geojson', data: cantonGeoJSON });
        
        map.addLayer({
          id: 'canton-fill',
          type: 'fill',
          source: 'cantons',
          paint: {
            'fill-color': '#A07CC5', // Default color
            'fill-opacity': 0.4
          }
        });
        
        // Border layer for all cantons
        map.addLayer({
          id: 'canton-borders',
          type: 'line',
          source: 'cantons',
          paint: {
            'line-color': '#000000',
            'line-width': 1
          }
        });
        
        // Border layer for selected canton
        map.addLayer({
          id: 'selected-canton-border',
          type: 'line',
          source: 'cantons',
          paint: {
            'line-color': '#FF0000', // Red for selected canton
            'line-width': 2
          },
          filter: ['==', 'NAME', ''] // Initially no canton is selected
        });
        
        
        // Highlight canton with a white border on hover
        map.addLayer({
          id: 'canton-highlight',
          type: 'line',
          source: 'cantons',
          paint: { 'line-color': '#fff', 'line-width': 3 },
          filter: ['==', 'NAME', '']
        });
        
        // Click Event
        const handleMapClick = (e) => {
          
          // If select same as previous canton, don't do anything
          // (we extract prev canton by getting the current selected-canton-border)
          if (e.features.length > 0 && e.features[0].properties.NAME != map.getFilter("selected-canton-border")[2]) {
            
            const cantonName = e.features[0].properties.NAME;
            const cantonBbox = bboxCache[cantonName];
            
            setClickedCanton(cantonName);
            
            // Show the red border only for the selected canton
            map.setFilter('selected-canton-border', ['==', 'NAME', cantonName]);
            
            // Determine the right padding based on which graph is selected
            let rightPadding = 50; // Default for collapsed sidebar
            
            
            if (graphExpandedRef.current === "Graph 3" || graphExpandedRef.current === "Graph 4") {
              rightPadding = 950; // Adjust for 900px width
            } else if (graphExpandedRef.current === "Graph 1" || graphExpandedRef.current === "Graph 2" || graphExpandedRef.current === "Volumes") {
              rightPadding = 650; // Adjust for 600px width
            } else {
              rightPadding = 350; // Default open sidebar
            }
            
            
            
            map.fitBounds(cantonBbox, {
              padding: { top: 50, bottom: 50, left: 50, right: rightPadding },
              maxZoom: 10,
              duration: 1000
            });
            
            if (graphExpandedRef.current === "Network" || graphExpandedRef.current === "Volumes") {
              loadNetworkForCanton(cantonName);
            } else {
              // Remove network-related layers and sources
              if (map.getLayer("network-layer")) {
                map.removeLayer("network-layer");
                map.removeLayer("click-network-layer");
                map.removeSource("network-source");
              }
              if (map.getLayer("ant-line")) {
                map.removeLayer("ant-line");
                map.removeSource("ant-path");
              }
              ["network-highlight"].forEach(id => {
                if (map.getLayer(id)) map.removeLayer(id);
                if (map.getSource(id)) map.removeSource(id);
              });
            }
          }
        };
        
        map.on('click', 'canton-fill', handleMapClick);
        
        return () => {
          map.off('click', 'canton-fill', handleMapClick);
          map.remove();
        };
      })
      .catch(error => console.error('Error loading GeoJSON:', error));
    }, [setClickedCanton]);
    
    // HANDLE HOVER
    useEffect(() => {
      let animationFrameId = null;
      
      const handleMouseMove = (e) => {
        if (animationFrameId) return;
        
        animationFrameId = requestAnimationFrame(() => {
          const features = mapRef.current.queryRenderedFeatures(e.point, {
            layers: ["canton-fill"]
          });
          
          if (features.length > 0) {
            const cantonName = features[0].properties.NAME;
            mapRef.current.setFilter("canton-highlight", ["==", "NAME", cantonName]);
          }
          
          animationFrameId = null;
        });
      };
      
      const handleMouseLeave = () => {
        mapRef.current.setFilter('canton-highlight', ['==', 'NAME', '']);
      };
      
      if (mapRef.current) {
        const map = mapRef.current;
        map.on('mousemove', 'canton-fill', handleMouseMove);
        map.on('mouseleave', 'canton-fill', handleMouseLeave);
        
        return () => {
          map.off('mousemove', 'canton-fill', handleMouseMove);
          map.off('mouseleave', 'canton-fill', handleMouseLeave);
        };
      }
    }, []);
    
    
    // ======================= HANDLING MATSIM NETWORK =======================
    
    const loadNetworkForCanton = (cantonName) => {
      
      const networkGeojsonPath = `/data/matsim/matsim_network_${cantonName}.geojson`;
      
      // Ensure the map is initialized
      if (!mapRef.current) return;
      const map = mapRef.current;
      
      // Remove layers from before (if applicable)
      const layersToRemove = ["network-layer", "click-network-layer", "ant-line", "network-highlight"];
      const sourcesToRemove = ["network-source", "network-highlight", "ant-path"];
      
      layersToRemove.forEach(id => {
        if (map.getLayer(id)) map.removeLayer(id);
      });
      
      sourcesToRemove.forEach(id => {
        if (map.getSource(id)) map.removeSource(id);
      });
      
      setIsLoadingNetwork(true); // show the loading screen
      setSelectedNetworkFeature(null); //reset selected segment
      
      // Check if the file exists before attempting to load it
      fetch(networkGeojsonPath, { method: "HEAD" })
      .then((response) => {
        if (!response.ok) {
          console.log(`No network file found for ${cantonName}, removing layer.`);
          return;
        }
        
        // Fetch the actual GeoJSON data
        return fetch(networkGeojsonPath).then((res) => res.json());
      })
      .then((networkGeojson) => {
        if (!networkGeojson) return; // Stop if no valid geojson was fetched
        
        map.addSource("network-source", { type: "geojson", data: networkGeojson });
        
        // add a invisible wider layer so easier to click
        map.addLayer({
          id: "click-network-layer",
          type: "line",
          source: "network-source",
          paint: {
            "line-width": [
              "interpolate", ["linear"], ["get", "capacity"],
              300, 7,   // Low volume → thin line
              4000, 14   // High volume → thick line
            ],
            "line-opacity": 0
          },
        });
        
        map.addLayer({
          id: "network-layer",
          type: "line",
          source: "network-source",
          paint: {
            "line-width": [
              "interpolate", ["linear"], ["get", "capacity"],
              300, 1,   // Low volume → thin line
              4000, 8   // High volume → thick line
            ],
            "line-color":
            graphExpandedRef.current === "Volumes"
            ? [
              "interpolate", ["linear"], ["get", "daily_avg_volume"],
              0, "#ffffcc",
              50, "#c2e699",
              100, "#78c679",
              250, "#31a354",
              500, "#006837"
            ]
            : [
              "interpolate", ["linear"], ["get", "freespeed"],
              0, "#ffffb2",
              6.94, "#fed976",
              13.89, "#feb24c",
              20.83, "#fd8d3c",
              27.78, "#fc4e2a",
              34.72, "#e31a1c",
              41.67, "#b10026",
            ],
          },
        });
        
        // Maintain current modes if changing cantons
        updateNetworkFilter(selectedNetworkModesRef.current);

        // only show 'car' roads for volumes
        if(graphExpandedRef.current === "Volumes"){
          map.setFilter("click-network-layer", ["match", ["index-of", "car", ["get", "modes"]], -1, false, true])
          map.setFilter("network-layer",["match", ["index-of", "car", ["get", "modes"]], -1, false, true])
        }
        
        // remove the loading screen when network finishes loading
        const handleIdle = () => {
          setIsLoadingNetwork(false);
          map.off("idle", handleIdle); // Clean up so it doesn't fire again unnecessarily
        };
        
        map.on("idle", handleIdle);
        
        // handle segment selection
        map.on("click", "click-network-layer", (e) => {
          if (!e.features.length) return;
          
          if (map.getLayer("ant-line")) {
            map.removeLayer("ant-line");
          }
          // Clean up old layers/sources
          ["network-highlight"].forEach(id => {
            if (map.getLayer(id)) map.removeLayer(id);
            if (map.getSource(id)) map.removeSource(id);
          });
          
          // Store selected features in list (ie if there are two overlapping segments)
          const feature_list = [e.features[0].properties];
          if (e.features[1]) {
            feature_list.push(e.features[1].properties);
          }
          
          // --------- Highlighting of clicked feature (for some reason this way runs fastest) -----------
          
          // get geojson of network-source
          const allFeatures = map.getSource("network-source")._data.features;
          
          // filter to ID in geojson
          const fullFeature = allFeatures.find(f => f.properties.id === feature_list[0].id);
          
          if (!fullFeature) return;
          
          const highlightGeoJSON = {
            type: "FeatureCollection",
            features: [fullFeature],
          };
          
          map.addSource("network-highlight", {
            type: "geojson",
            data: highlightGeoJSON,
          });
          
          map.addLayer({
            id: "network-highlight",
            type: "line",
            source: "network-highlight",
            paint: {
              "line-width": [
                "interpolate", ["linear"], ["get", "capacity"],
                300, 5,
                4000, 14,
              ],
              "line-color": "#8affff",
              "line-opacity": 1,
            },
          }, "network-layer"); 
          
          // Store and display feature attributes
          setSelectedNetworkFeature(feature_list);
        });
      })
      .catch((error) =>
        console.error(`Error loading network for ${cantonName}:`, error)
    );
  };
  
  // --------- UPDATING NETWORK BY MODE ----------
  const updateNetworkFilter = (modes) => {
    if (!mapRef.current) return;
    const map = mapRef.current;
    
    if (!map.getLayer("network-layer")) return;
    
    // If "all" modes selected, remove filter
    if (!modes || modes.includes("all")) {
      map.setFilter("network-layer", null);
      map.setFilter("click-network-layer", null);
      if (map.getLayer("network-highlight")){
        map.setFilter("network-highlight", null);
      }
    } else {
      // set filter for roads that match ANY of the selected modes
      map.setFilter("network-layer", [
        "any",
        ...modes.map(mode => ["match", ["index-of", mode, ["get", "modes"]], -1, false, true])
      ]);
      map.setFilter("click-network-layer", [
        "any",
        ...modes.map(mode => ["match", ["index-of", mode, ["get", "modes"]], -1, false, true])
      ]);
      
      // if the highlight also exists, filter that too
      if (map.getLayer("network-highlight")){
        map.setFilter("network-highlight", [
          "any",
          ...modes.map(mode => ["match", ["index-of", mode, ["get", "modes"]], -1, false, true])
        ]);
      }
      
      // We don't need to filter the ant-path because it will not
      // appear on the same screen as the matsim filter screen
    }
  };
  
  // Update network mode filter (MatSIM Network) on change
  useEffect(() => {
    if (!mapRef.current) return;

    selectedNetworkModesRef.current = selectedNetworkModes;
    updateNetworkFilter(selectedNetworkModes); // Apply mode filter when it changes
  }, [selectedNetworkModes]);
  
  // ---------------- ADD ANT PATH TO VISUALIZE VOLUME DIRECTION ----------------
  useEffect(() => {
    if (!visualizeLinkId || !mapRef.current) return;
    
    const map = mapRef.current;
    const source = map.getSource("network-source");
    
    if (!source || !source._data) return;
    
    // Access full GeoJSON source
    const fullGeoJSON = source._data;
    const feature = fullGeoJSON.features.find(f => f.properties.id === visualizeLinkId);
    
    if (!feature) return;
    
    // convert MultiLineString into a LineString
    // if a line is discontinuous, it forces it to be continuous
    // ie: -----      ------- →  --------------------
    const mergedCoords =
    feature.geometry.type === "LineString"
    ? feature.geometry.coordinates
    : feature.geometry.type === "MultiLineString"
    ? feature.geometry.coordinates.flat()
    : [];
    
    if (mergedCoords.length < 2) return;
    
    // Remove existing layer/source
    if (map.getLayer("ant-line")) map.removeLayer("ant-line");
    if (map.getSource("ant-path")) map.removeSource("ant-path");
    
    // Add new source with a single LineString
    map.addSource("ant-path", {
      type: "geojson",
      data: {
        type: "Feature",
        geometry: {
          type: "LineString",
          coordinates: mergedCoords,
        },
        properties: {},
      },
    });
    
    map.addLayer({
      id: "ant-line",
      type: "line",
      source: "ant-path",
      layout: {},
      paint: {
        "line-color": "#FF00FF",
        "line-width": 4,
        "line-dasharray": [3, 3], // initial
      },
    });
    
    // Create dash animation sequence
    const dashArraySeq = [
      [0, 0.3, 3, 2.7],
      [0, 0.6, 3, 2.4],
      [0, 0.9, 3, 2.1],
      [0, 1.2, 3, 1.8],
      [0, 1.5, 3, 1.5],
      [0, 1.8, 3, 1.2],
      [0, 2.1, 3, 0.9],
      [0, 2.4, 3, 0.6],
      [0, 2.7, 3, 0.3],
      [0, 3.0, 3, 0],
      [0.3, 3, 2.7, 0],
      [0.6, 3, 2.4, 0],
      [0.9, 3, 2.1, 0],
      [1.2, 3, 1.8, 0],
      [1.5, 3, 1.5, 0],
      [1.8, 3, 1.2, 0],
      [2.1, 3, 0.9, 0],
      [2.4, 3, 0.6, 0],
      [2.7, 3, 0.3, 0],
      [3, 3, 0, 0],
    ];
    let dashArrayIdx = 0;
    let lastUpdateTime = 0;
    const frameIntervalMs = 50; // update every 50ms
    
    // Animate line as a "ant path", by constantly changing the dash-array sequence
    // to make it appear to "move"
    function animateLine(timestamp) {
      if (!map.getLayer("ant-line")) return;
      
      if (timestamp - lastUpdateTime >= frameIntervalMs) {
        dashArrayIdx = (dashArrayIdx + 1) % dashArraySeq.length;
        map.setPaintProperty("ant-line", "line-dasharray", dashArraySeq[dashArrayIdx]);
        lastUpdateTime = timestamp;
      }
      
      requestAnimationFrame(animateLine);
    }
    
    requestAnimationFrame(animateLine);
    
    return () => {
      if (map.getLayer("ant-line")) map.removeLayer("ant-line");
      if (map.getSource("ant-path")) map.removeSource("ant-path");
    };
  }, [visualizeLinkId]); // run when visualizeLinkId changes
  
  // handles switching between network / non-network modules on the sidebar
  useEffect(() => {
    const map = mapRef.current;
    const canton = searchCanton;
    
    graphExpandedRef.current = isGraphExpanded;
    
    if (!map || !canton) return;
    
    // Hides all network layers
    const hideNetworkLayers = () => {
      ["network-layer", "click-network-layer", "network-highlight"].forEach(id => {
        if (map.getLayer(id)) {
          map.setLayoutProperty(id, "visibility", "none");
        }
      });
    };
    
    // Shows network layers
    const showNetworkLayers = () => {
      ["network-layer", "click-network-layer", "network-highlight"].forEach(id => {
        if (map.getLayer(id)) {
          map.setLayoutProperty(id, "visibility", "visible");
        }
      });
    };
    
    if (isGraphExpanded === "Network" || isGraphExpanded === "Volumes") {
      if (map.getLayer("network-layer")) {
        showNetworkLayers(); 
      } else {
        loadNetworkForCanton(canton);
      }
    } else {
      hideNetworkLayers();
    }
    
    if (!map.getLayer("network-layer")) return;
    
    // Update line color based on selected module
    const colorRamp = isGraphExpanded === "Volumes"
    ? [
      "interpolate", ["linear"], ["get", "daily_avg_volume"],
      0, "#ffffcc",
      50, "#c2e699",
      100, "#78c679",
      250, "#31a354",
      500, "#006837"
    ]
    : [
      "interpolate", ["linear"], ["get", "freespeed"],
      0, "#ffffb2",
      6.94, "#fed976",
      13.89, "#feb24c",
      20.83, "#fd8d3c",
      27.78, "#fc4e2a",
      34.72, "#e31a1c",
      41.67, "#b10026"
    ];
    
    map.setPaintProperty("network-layer", "line-color", colorRamp);
    
    if (map.getLayer("ant-line")) {
      map.removeLayer("ant-line");
    }
    
    // Handles maintaining highlight when swapping between network / volumes
    if (map.getSource("network-highlight")) {
      const source = map.getSource("network-highlight");
      
      // check if the highlight contains mode "car" from its source
      let hasCarMode = false;
      
      if (source && source._data) {
        const features = source._data.features;
        
        hasCarMode = features.some(f => {
          const modes = f.properties?.modes;
          return modes?.split(",").includes("car");
        });
      }
      
      // if not a car, remove it
      if (!hasCarMode) {
        setSelectedNetworkFeature(null);
        
        // if its not a car + we swap back to network, retrieve the
        // segment properties from "network-highlight" source
        if (isGraphExpanded === "Network"){
          setSelectedNetworkFeature([source._data.features[0].properties]);
        }
      }
    }
    
  }, [isGraphExpanded]);
  
  // ======================= CHOROPLETH MODULE =======================
  
  useEffect(() => {
    fetch("/data/canton_mode_share.json")
    .then((response) => response.json())
    .then((data) => {
      setModeShareData(data);
      setMaxSharePerMode(data.max_share_per_mode);
    })
    .catch((error) => console.error("Error loading mode share data:", error));
  }, []);

  // Set colours for choropleth by mode (matches with plots)
  const MODE_COLORS = {
    car: "#636efa",
    car_passenger: "#ef553b",
    pt: "#00cc96",
    bike: "#ab63fa",
    walk: "#ffa15a",
  };
  
  // Set choropleth colours for cantons
  useEffect(() => {
    const map = mapRef.current;
    
    if (!map.getLayer("canton-fill")) return; // incase canton-fill not loaded yet
    
    if (selectedMode === "None") {
      // Reset to default
      map.setPaintProperty("canton-fill", "fill-opacity", 0.4);
      map.setPaintProperty("canton-fill", "fill-color", "#A07CC5");
      return;
    }
    
    const maxShare = maxSharePerMode[selectedMode] || 1;
    
    const colorStops = modeShareData[selectedDataset]
    .filter(entry => entry.mode === selectedMode)
    .reduce((acc, entry) => {
      const normalizedShare = entry.share / maxShare;
      acc[entry.canton_name] = `rgb(${interpolateColor("#FFFFFF", MODE_COLORS[selectedMode], normalizedShare)})`; // White → Full Color
      return acc;
    }, {});
    
    map.setPaintProperty("canton-fill", "fill-color", [
      "case",
      ...Object.entries(colorStops).flatMap(([canton, color]) => [["==", ["get", "NAME"], canton], color]),
      "#FFFFFF"
    ]);
    
    map.setPaintProperty("canton-fill", "fill-opacity", 1.0); // No transparency
    
  }, [modeShareData, selectedMode, selectedDataset, maxSharePerMode]);
  
  // Function to interpolate between two colors (White → Mode Color)
  const interpolateColor = (color1, color2, factor) => {
    const rgb1 = hexToRgb(color1);
    const rgb2 = hexToRgb(color2);
    
    return rgb1.map((c, i) => Math.round(c + (rgb2[i] - c) * factor)).join(", ");
  };
  
  // Convert HEX to RGB array (since we defined colours in HEX)
  const hexToRgb = (hex) => {
    const bigint = parseInt(hex.slice(1), 16);
    return [(bigint >> 16) & 255, (bigint >> 8) & 255, bigint & 255];
  };
  
  //  ======================= MAP PADDING LOGIC =======================
  // this ensures that the map stays focused on the same feature even
  // as the width of the sidebar changes (otherwise as the sidebar)
  // gets wider, it could hide what we are originally looking at

  useEffect(() => {
    if (mapRef.current) {
      const map = mapRef.current;
      
      // Determine the right padding based on which graph is selected
      let rightPadding = 50; // Default for collapsed sidebar
      
      if (isSidebarOpen) {
        // Widest width
        if (isGraphExpanded === "Graph 3" || isGraphExpanded === "Graph 4") {
          rightPadding = 950; // Adjust for 900px width
        // Middle width
        } else if (isGraphExpanded === "Graph 1" || isGraphExpanded === "Graph 2" || isGraphExpanded === "Volumes") {
          rightPadding = 650; // Adjust for 600px width
        } else {
        // Smallest width
          rightPadding = 350;
        }
      }
      
      // Smoothly adjust padding when sidebar changes
      map.easeTo({
        padding: { top: 50, bottom: 50, left: 50, right: rightPadding },
        duration: 600,
      });
    }
  }, [isSidebarOpen, isGraphExpanded]); // Re-run when sidebar size changes
  
  // handle search-based zooming
  useEffect(() => {
    if (searchCanton && mapRef.current && bboxCache[searchCanton]) {
      const map = mapRef.current;
      const cantonBbox = bboxCache[searchCanton]; // fetch bbox
      
      // Determine the correct right padding
      let rightPadding = 50; // Default for collapsed sidebar
      if (isSidebarOpen) {
        if (isGraphExpanded === "Graph 3" || isGraphExpanded === "Graph 4") {
          rightPadding = 950; // Adjust for 900px width
        } else if (isGraphExpanded === "Graph 1" || isGraphExpanded === "Graph 2" || isGraphExpanded === "Volumes") {
          rightPadding = 650; // Adjust for 600px width
        } else {
          rightPadding = 350; // Default open sidebar
        }
      }
      
      map.fitBounds(cantonBbox, {
        padding: { top: 50, bottom: 50, left: 50, right: rightPadding },
        maxZoom: 10,
        duration: 1000,
      });
      
      setClickedCanton(searchCanton);
      
      map.setFilter("selected-canton-border", ["==", "NAME", searchCanton]);
      
      if (graphExpandedRef.current === "Network" || graphExpandedRef.current === "Volumes") {
        loadNetworkForCanton(searchCanton);
      } else {
        // Remove network-related layers and sources
        if (map.getLayer("network-layer")) {
          map.removeLayer("network-layer");
          map.removeLayer("click-network-layer");
          map.removeSource("network-source");
        }
        if (map.getLayer("ant-line")) {
          map.removeLayer("ant-line");
          map.removeSource("ant-path")
        }
        ["network-highlight"].forEach(id => {
          if (map.getLayer(id)) map.removeLayer(id);
          if (map.getSource(id)) map.removeSource(id);
        });
      }
    }
  }, [searchCanton]); // only update when searchCanton updates
  
  return (
    <>
    <div id="map-container" ref={mapContainerRef} />
    
    {isLoadingNetwork && (
      <div className="map-loading-overlay">
      <div className="spinner" />
      <div className="loading-text">Loading network...</div>
      </div>
    )}
    </>
  );
};

export default Map;