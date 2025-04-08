import { useState } from 'react';
import './CantonSearch.css';

const CantonSearch = ({ cantonList, onSearch }) => {
    const [searchTerm, setSearchTerm] = useState(''); // search term
    const [filteredCantons, setFilteredCantons] = useState([]); // cantons which match term
    const [selectedIndex, setSelectedIndex] = useState(-1); // selected item in list

    const normalizeString = (str) => {
        return str
            .normalize("NFD") // Decomposes accents
            .replace(/[\u0300-\u036f]/g, "") // Removes accents
            .toLowerCase(); // Converts to lowercase
    };

    const handleSearch = (cantonName = null) => {
        const normalizedSearch = normalizeString(cantonName || searchTerm);

        let matchedCanton = cantonList.find(
            (canton) => normalizeString(canton) === normalizedSearch
        );

        // If no exact match, use the first suggestion
        if (!matchedCanton && filteredCantons.length > 0) {
            matchedCanton = filteredCantons[0];
        }

        if (matchedCanton) {
            setSearchTerm(''); // Clear input after selection
            onSearch(matchedCanton);
            setFilteredCantons([]); // Clear suggestions
            setSelectedIndex(-1); // Reset selection
        }
    };

    const handleInputChange = (e) => {
        const input = e.target.value;
        setSearchTerm(input);
        setSelectedIndex(-1); // Reset selection when typing
    
        if (input.trim() === '') {
            setFilteredCantons([]);
            return;
        }
    
        const normalizedInput = normalizeString(input);
    
        const startsWithMatches = [];
        const containsMatches = [];
    
        // Separate cantons into "starts with" and "contains"
        cantonList.forEach(canton => {
            const normalizedCanton = normalizeString(canton);
            if (normalizedCanton.startsWith(normalizedInput)) {
                startsWithMatches.push(canton);
            } else if (normalizedCanton.includes(normalizedInput)) {
                containsMatches.push(canton);
            }
        });
    
        // Sort both groups alphabetically and merge them
        const suggestions = [
            ...startsWithMatches.sort((a, b) => a.localeCompare(b)),
            ...containsMatches.sort((a, b) => a.localeCompare(b))
        ].slice(0, 5); // Limit suggestions to 5
    
        setFilteredCantons(suggestions);
    };
    

    const handleKeyDown = (e) => {
        if (e.key === 'ArrowDown') {
            setSelectedIndex((prevIndex) => Math.min(prevIndex + 1, filteredCantons.length - 1));
        } else if (e.key === 'ArrowUp') {
            setSelectedIndex((prevIndex) => Math.max(prevIndex - 1, 0));
        } else if (e.key === 'Enter') {
            if (selectedIndex >= 0 && filteredCantons.length > 0) {
                handleSearch(filteredCantons[selectedIndex]);
            } else if (filteredCantons.length > 0) {
                handleSearch(filteredCantons[0]); // Select the first suggestion if no selection
            }
        }
    };

    return (
        <div className="canton-search">
            <input
                type="text"
                value={searchTerm}
                onChange={handleInputChange}
                onKeyDown={handleKeyDown}
                placeholder="Search Canton..."
                className="canton-input"
            />
            <button onClick={() => handleSearch()} className="search-button">Search</button>

            {filteredCantons.length > 0 && (
                <ul className="suggestions">
                    {filteredCantons.map((canton, index) => (
                        <li
                            key={index}
                            onClick={() => handleSearch(canton)}
                            className={index === selectedIndex ? "selected" : ""}
                        >
                            {canton}
                        </li>
                    ))}
                </ul>
            )}
        </div>
    );
};

export default CantonSearch;
