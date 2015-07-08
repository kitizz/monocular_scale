Figure {
    id: figure
    Axis {
        handle: "ax"
        LinePlot {
            handle: "bluePlot"
            line { color: "#589BDA"; style: "-"; width: 3 }
        }
        LinePlot {
            handle: "redPlot"
            line { color: "#EF5762"; style: "--"; width: 3 }
        }
        LinePlot {
            handle: "greenPlot"
            line { color: "#66FF66"; style: ":"; width: 3 }
        }
        LinePlot {
            handle: "greenDots"
            line { color: "#66FF66"; style: "."; width: 6 }
        }
    }
}