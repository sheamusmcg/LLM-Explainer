/* Shared utilities for D3 visualization components */

const VizUtils = {
    // Token color palette (20 distinct colors for token highlighting)
    TOKEN_COLORS: [
        '#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9', '#BAE1FF',
        '#E8BAFF', '#FFB3E6', '#B3FFE6', '#FFD9B3', '#B3D9FF',
        '#D9FFB3', '#FFB3D9', '#B3FFD9', '#D9B3FF', '#FFE6B3',
        '#B3FFB3', '#E6B3FF', '#B3E6FF', '#FFB3B3', '#B3FFFF',
    ],

    // Get color for a token by index
    tokenColor(index) {
        return this.TOKEN_COLORS[index % this.TOKEN_COLORS.length];
    },

    // Blue sequential color scale for attention weights
    attentionColorScale: d3.scaleSequential(d3.interpolateBlues).domain([0, 1]),

    // Diverging color scale for logits
    logitColorScale: d3.scaleSequential(d3.interpolateRdYlGn).domain([-5, 5]),

    // Probability color scale (green gradient)
    probColorScale(p) {
        const intensity = Math.min(p * 5, 1);  // Scale up small probs for visibility
        return d3.interpolateGreens(0.2 + intensity * 0.8);
    },

    // Confidence color (green = high, red = low)
    confidenceColor(p) {
        if (p > 0.5) return '#2ecc71';
        if (p > 0.2) return '#f39c12';
        if (p > 0.05) return '#e67e22';
        return '#e74c3c';
    },

    // Format a probability as percentage
    formatProb(p) {
        if (p >= 0.01) return (p * 100).toFixed(1) + '%';
        if (p >= 0.001) return (p * 100).toFixed(2) + '%';
        return (p * 100).toFixed(3) + '%';
    },

    // Format a token for display (make whitespace visible)
    formatToken(t) {
        return t.replace(/ /g, '\u00B7').replace(/\n/g, '\u21B5');
    },

    // Read component data from the JSON script tag
    getData() {
        const el = document.getElementById('component-data');
        return JSON.parse(el.textContent);
    },

    // Create a tooltip div
    createTooltip() {
        return d3.select('body')
            .append('div')
            .attr('class', 'viz-tooltip')
            .style('opacity', 0);
    },

    // Show tooltip
    showTooltip(tooltip, html, event) {
        tooltip
            .html(html)
            .style('opacity', 1)
            .style('left', (event.pageX + 12) + 'px')
            .style('top', (event.pageY - 10) + 'px');
    },

    // Hide tooltip
    hideTooltip(tooltip) {
        tooltip.style('opacity', 0);
    },
};
