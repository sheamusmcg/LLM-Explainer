/* Tokenizer visualization: colored token spans with IDs */
(function() {
    const data = VizUtils.getData();
    const container = d3.select('#viz-container');

    // Stats bar
    const stats = container.append('div').attr('class', 'stats-bar');
    stats.append('div').attr('class', 'stat-item')
        .html(`Tokens: <span class="stat-value">${data.tokens.length}</span>`);
    stats.append('div').attr('class', 'stat-item')
        .html(`Characters: <span class="stat-value">${data.text.length}</span>`);
    stats.append('div').attr('class', 'stat-item')
        .html(`Ratio: <span class="stat-value">${(data.text.length / data.tokens.length).toFixed(1)} chars/token</span>`);
    stats.append('div').attr('class', 'stat-item')
        .html(`Vocabulary: <span class="stat-value">${data.vocab_size.toLocaleString()} tokens</span>`);

    // Token container
    const tokenContainer = container.append('div').attr('class', 'token-container');

    const tooltip = VizUtils.createTooltip();

    data.tokens.forEach((token, i) => {
        const color = VizUtils.tokenColor(i);
        const tokenEl = tokenContainer.append('div')
            .attr('class', 'token')
            .style('background-color', color)
            .on('mouseover', function(event) {
                VizUtils.showTooltip(tooltip,
                    `<strong>Token ${i + 1}</strong><br>` +
                    `Text: "${VizUtils.formatToken(token)}"<br>` +
                    `ID: ${data.token_ids[i]}<br>` +
                    `Length: ${token.length} chars`,
                    event
                );
            })
            .on('mousemove', function(event) {
                tooltip
                    .style('left', (event.pageX + 12) + 'px')
                    .style('top', (event.pageY - 10) + 'px');
            })
            .on('mouseout', function() {
                VizUtils.hideTooltip(tooltip);
            });

        tokenEl.append('span')
            .attr('class', 'token-text')
            .text(VizUtils.formatToken(token));

        tokenEl.append('span')
            .attr('class', 'token-id')
            .text(data.token_ids[i]);
    });

    // Legend
    container.append('div')
        .style('margin-top', '12px')
        .style('font-size', '12px')
        .style('color', '#888')
        .text('Each colored block is one token. Hover for details. The number below each token is its vocabulary ID.');
})();
