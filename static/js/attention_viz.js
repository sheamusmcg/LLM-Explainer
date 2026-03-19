/* Attention visualization: heatmap and arc diagram */
(function() {
    const data = VizUtils.getData();
    const container = d3.select('#viz-container');
    const tokens = data.tokens;
    const weights = data.weights;
    const threshold = data.threshold;
    const n = tokens.length;

    const tooltip = VizUtils.createTooltip();

    if (data.view === 'heatmap') {
        renderHeatmap();
    } else {
        renderArcs();
    }

    function renderHeatmap() {
        const margin = {top: 80, right: 20, bottom: 20, left: 80};
        const cellSize = Math.min(40, Math.max(20, 500 / n));
        const width = margin.left + n * cellSize + margin.right;
        const height = margin.top + n * cellSize + margin.bottom;

        const svg = container.append('svg')
            .attr('width', width)
            .attr('height', height);

        const g = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        // Color scale
        const maxVal = d3.max(weights.flat());
        const colorScale = d3.scaleSequential(d3.interpolateBlues)
            .domain([0, maxVal]);

        // Draw cells
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                const val = weights[i][j];
                if (val < threshold) continue;

                g.append('rect')
                    .attr('class', 'heatmap-cell')
                    .attr('x', j * cellSize)
                    .attr('y', i * cellSize)
                    .attr('width', cellSize)
                    .attr('height', cellSize)
                    .attr('fill', colorScale(val))
                    .attr('rx', 2)
                    .on('mouseover', function(event) {
                        d3.select(this).attr('stroke', '#333').attr('stroke-width', 2);
                        VizUtils.showTooltip(tooltip,
                            `<strong>"${tokens[i]}"</strong> attends to <strong>"${tokens[j]}"</strong><br>` +
                            `Weight: ${val.toFixed(4)} (${VizUtils.formatProb(val)})`,
                            event
                        );
                    })
                    .on('mousemove', function(event) {
                        tooltip.style('left', (event.pageX + 12) + 'px').style('top', (event.pageY - 10) + 'px');
                    })
                    .on('mouseout', function() {
                        d3.select(this).attr('stroke', '#fff').attr('stroke-width', 1);
                        VizUtils.hideTooltip(tooltip);
                    });
            }
        }

        // Row labels (query tokens — "from")
        for (let i = 0; i < n; i++) {
            g.append('text')
                .attr('class', 'axis-label')
                .attr('x', -6)
                .attr('y', i * cellSize + cellSize / 2)
                .attr('text-anchor', 'end')
                .attr('dominant-baseline', 'middle')
                .text(tokens[i].length > 8 ? tokens[i].slice(0, 7) + '…' : tokens[i]);
        }

        // Column labels (key tokens — "to")
        for (let j = 0; j < n; j++) {
            g.append('text')
                .attr('class', 'axis-label')
                .attr('x', j * cellSize + cellSize / 2)
                .attr('y', -6)
                .attr('text-anchor', 'start')
                .attr('dominant-baseline', 'middle')
                .attr('transform', `rotate(-45, ${j * cellSize + cellSize / 2}, -6)`)
                .text(tokens[j].length > 8 ? tokens[j].slice(0, 7) + '…' : tokens[j]);
        }

        // Axis titles
        svg.append('text')
            .attr('x', margin.left / 2)
            .attr('y', margin.top + n * cellSize / 2)
            .attr('text-anchor', 'middle')
            .attr('transform', `rotate(-90, ${margin.left / 2 - 20}, ${margin.top + n * cellSize / 2})`)
            .style('font-size', '12px')
            .style('fill', '#888')
            .text('Query (from)');

        svg.append('text')
            .attr('x', margin.left + n * cellSize / 2)
            .attr('y', 14)
            .attr('text-anchor', 'middle')
            .style('font-size', '12px')
            .style('fill', '#888')
            .text('Key (to)');

        // Color legend
        const legendWidth = 120;
        const legendG = svg.append('g')
            .attr('transform', `translate(${margin.left + n * cellSize - legendWidth}, ${margin.top - 40})`);

        const defs = svg.append('defs');
        const gradient = defs.append('linearGradient')
            .attr('id', 'heatmap-gradient');
        gradient.append('stop').attr('offset', '0%').attr('stop-color', colorScale(0));
        gradient.append('stop').attr('offset', '100%').attr('stop-color', colorScale(maxVal));

        legendG.append('rect')
            .attr('width', legendWidth)
            .attr('height', 10)
            .attr('rx', 3)
            .style('fill', 'url(#heatmap-gradient)');

        legendG.append('text')
            .attr('x', 0).attr('y', -3)
            .style('font-size', '10px').style('fill', '#888')
            .text('0');
        legendG.append('text')
            .attr('x', legendWidth).attr('y', -3)
            .attr('text-anchor', 'end')
            .style('font-size', '10px').style('fill', '#888')
            .text(maxVal.toFixed(2));
    }

    function renderArcs() {
        const margin = {top: 20, right: 40, bottom: 120, left: 40};
        const tokenSpacing = Math.min(70, Math.max(40, 700 / n));
        const width = margin.left + n * tokenSpacing + margin.right;
        const arcAreaHeight = 180;
        const height = margin.top + arcAreaHeight + margin.bottom;

        const svg = container.append('svg')
            .attr('width', width)
            .attr('height', height);

        const g = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        const tokenY = arcAreaHeight;
        const colorScale = d3.scaleSequential(d3.interpolateBlues).domain([0, 1]);

        // Draw arcs (query → key)
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                if (i === j) continue;
                const val = weights[i][j];
                if (val < threshold) continue;

                const x1 = i * tokenSpacing + tokenSpacing / 2;
                const x2 = j * tokenSpacing + tokenSpacing / 2;
                const midX = (x1 + x2) / 2;
                const arcHeight = Math.abs(x2 - x1) * 0.4;

                g.append('path')
                    .attr('class', 'attention-arc')
                    .attr('d', `M ${x1} ${tokenY} Q ${midX} ${tokenY - arcHeight} ${x2} ${tokenY}`)
                    .attr('stroke', colorScale(val))
                    .attr('stroke-width', Math.max(1, val * 6))
                    .on('mouseover', function(event) {
                        d3.select(this).attr('stroke-opacity', 0.9).attr('stroke-width', Math.max(2, val * 8));
                        VizUtils.showTooltip(tooltip,
                            `<strong>"${tokens[i]}"</strong> → <strong>"${tokens[j]}"</strong><br>` +
                            `Attention: ${VizUtils.formatProb(val)}`,
                            event
                        );
                    })
                    .on('mousemove', function(event) {
                        tooltip.style('left', (event.pageX + 12) + 'px').style('top', (event.pageY - 10) + 'px');
                    })
                    .on('mouseout', function() {
                        d3.select(this).attr('stroke-opacity', 0.4).attr('stroke-width', Math.max(1, val * 6));
                        VizUtils.hideTooltip(tooltip);
                    });
            }
        }

        // Draw token labels
        for (let i = 0; i < n; i++) {
            const x = i * tokenSpacing + tokenSpacing / 2;

            g.append('text')
                .attr('class', 'arc-token')
                .attr('x', x)
                .attr('y', tokenY + 18)
                .text(tokens[i].length > 8 ? tokens[i].slice(0, 7) + '…' : tokens[i])
                .attr('transform', `rotate(45, ${x}, ${tokenY + 18})`)
                .on('mouseover', function(event) {
                    // Highlight all arcs from this token
                    g.selectAll('.attention-arc').attr('stroke-opacity', 0.05);
                    // Re-highlight arcs from this token (index i)
                    let arcIdx = 0;
                    for (let qi = 0; qi < n; qi++) {
                        for (let kj = 0; kj < n; kj++) {
                            if (qi === kj) continue;
                            if (weights[qi][kj] < threshold) continue;
                            if (qi === i) {
                                g.selectAll('.attention-arc').filter(function(d, idx) { return idx === arcIdx; })
                                    .attr('stroke-opacity', 0.8);
                            }
                            arcIdx++;
                        }
                    }
                })
                .on('mouseout', function() {
                    g.selectAll('.attention-arc').attr('stroke-opacity', 0.4);
                });

            // Token index
            g.append('text')
                .attr('x', x)
                .attr('y', tokenY + 42)
                .attr('text-anchor', 'middle')
                .style('font-size', '9px')
                .style('fill', '#aaa')
                .attr('transform', `rotate(45, ${x}, ${tokenY + 42})`)
                .text(i);
        }
    }
})();
