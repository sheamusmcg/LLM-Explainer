/* Output visualization: horizontal bar chart of token probabilities */
(function() {
    const data = VizUtils.getData();
    const container = d3.select('#viz-container');
    const tokens = data.top_tokens;

    const margin = {top: 30, right: 80, bottom: 20, left: 120};
    const barHeight = 20;
    const barGap = 2;
    const width = 700;
    const height = margin.top + tokens.length * (barHeight + barGap) + margin.bottom;

    const svg = container.append('svg')
        .attr('width', width)
        .attr('height', height);

    const g = svg.append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    const maxProb = d3.max(tokens, d => d.probability);
    const xScale = d3.scaleLinear()
        .domain([0, maxProb])
        .range([0, width - margin.left - margin.right]);

    const tooltip = VizUtils.createTooltip();

    // Title
    svg.append('text')
        .attr('x', width / 2)
        .attr('y', 18)
        .attr('text-anchor', 'middle')
        .style('font-size', '13px')
        .style('fill', '#888')
        .text(`Next token probabilities (temperature = ${data.temperature})`);

    tokens.forEach((token, i) => {
        const y = i * (barHeight + barGap);
        const barWidth = xScale(token.probability);

        // Bar
        g.append('rect')
            .attr('class', 'bar')
            .attr('x', 0)
            .attr('y', y)
            .attr('width', barWidth)
            .attr('height', barHeight)
            .attr('fill', i === 0 ? '#4A90D9' : VizUtils.probColorScale(token.probability))
            .attr('rx', 3)
            .on('mouseover', function(event) {
                d3.select(this).attr('opacity', 0.8);
                VizUtils.showTooltip(tooltip,
                    `<strong>"${VizUtils.formatToken(token.token)}"</strong><br>` +
                    `Probability: ${VizUtils.formatProb(token.probability)}<br>` +
                    `Rank: #${i + 1}`,
                    event
                );
            })
            .on('mousemove', function(event) {
                tooltip.style('left', (event.pageX + 12) + 'px').style('top', (event.pageY - 10) + 'px');
            })
            .on('mouseout', function() {
                d3.select(this).attr('opacity', 1);
                VizUtils.hideTooltip(tooltip);
            });

        // Token label (left)
        g.append('text')
            .attr('class', 'bar-label')
            .attr('x', -6)
            .attr('y', y + barHeight / 2)
            .attr('text-anchor', 'end')
            .attr('dominant-baseline', 'middle')
            .style('font-weight', i === 0 ? '700' : '400')
            .text(VizUtils.formatToken(token.token));

        // Probability label (right)
        g.append('text')
            .attr('class', 'bar-value')
            .attr('x', barWidth + 6)
            .attr('y', y + barHeight / 2)
            .attr('dominant-baseline', 'middle')
            .text(VizUtils.formatProb(token.probability));
    });

    // Highlight indicator for top prediction
    g.append('text')
        .attr('x', xScale(tokens[0].probability) + 50)
        .attr('y', barHeight / 2)
        .attr('dominant-baseline', 'middle')
        .style('font-size', '11px')
        .style('fill', '#4A90D9')
        .style('font-weight', '600')
        .text('← most likely');
})();
