/* Generation visualization: token stream with confidence coloring */
(function() {
    const data = VizUtils.getData();
    const container = d3.select('#viz-container');

    // Header info
    container.append('div')
        .attr('class', 'stats-bar')
        .html(
            `<div class="stat-item">Strategy: <span class="stat-value">${data.strategy}</span></div>` +
            `<div class="stat-item">Temperature: <span class="stat-value">${data.temperature}</span></div>` +
            `<div class="stat-item">Tokens generated: <span class="stat-value">${data.steps.length}</span></div>`
        );

    // Token stream
    const stream = container.append('div')
        .style('margin', '16px 0')
        .style('line-height', '2.2')
        .style('font-size', '15px');

    const tooltip = VizUtils.createTooltip();

    // Prompt tokens (gray)
    const promptWords = data.prompt.split(/(\s+)/);
    promptWords.forEach(word => {
        if (word.trim()) {
            stream.append('span')
                .attr('class', 'gen-token prompt')
                .text(word);
        } else if (word) {
            stream.append('span').text(word);
        }
    });

    // Generated tokens (colored by confidence)
    data.steps.forEach((step, i) => {
        const color = VizUtils.confidenceColor(step.probability);
        const bgColor = color + '22';  // Very transparent
        const borderColor = color + '66';

        stream.append('span')
            .attr('class', 'gen-token generated')
            .style('background-color', bgColor)
            .style('border-color', borderColor)
            .style('border', `1px solid ${borderColor}`)
            .style('color', '#262730')
            .text(VizUtils.formatToken(step.token))
            .on('mouseover', function(event) {
                d3.select(this).style('transform', 'translateY(-2px)')
                    .style('box-shadow', `0 4px 12px ${color}44`);

                let altHtml = step.alternatives.slice(0, 5).map((alt, j) => {
                    const isSelected = alt.token === step.token;
                    const marker = isSelected ? ' ✓' : '';
                    return `${VizUtils.formatToken(alt.token)}: ${VizUtils.formatProb(alt.probability)}${marker}`;
                }).join('<br>');

                VizUtils.showTooltip(tooltip,
                    `<strong>Step ${step.step}</strong><br>` +
                    `Token: "${VizUtils.formatToken(step.token)}"<br>` +
                    `Confidence: ${VizUtils.formatProb(step.probability)}<br>` +
                    `<br><strong>Top alternatives:</strong><br>${altHtml}`,
                    event
                );
            })
            .on('mousemove', function(event) {
                tooltip.style('left', (event.pageX + 12) + 'px').style('top', (event.pageY - 10) + 'px');
            })
            .on('mouseout', function() {
                d3.select(this).style('transform', 'none').style('box-shadow', 'none');
                VizUtils.hideTooltip(tooltip);
            });
    });

    // Legend
    const legend = container.append('div')
        .attr('class', 'legend')
        .style('margin-top', '12px');

    legend.append('span').text('Confidence:');

    const colors = [
        {label: 'High (>50%)', color: '#2ecc71'},
        {label: 'Medium (>20%)', color: '#f39c12'},
        {label: 'Low (>5%)', color: '#e67e22'},
        {label: 'Very low (<5%)', color: '#e74c3c'},
    ];

    colors.forEach(c => {
        const item = legend.append('span').style('display', 'inline-flex').style('align-items', 'center').style('gap', '4px');
        item.append('span')
            .style('width', '12px').style('height', '12px')
            .style('border-radius', '3px')
            .style('background-color', c.color)
            .style('display', 'inline-block');
        item.append('span').text(c.label);
    });
})();
