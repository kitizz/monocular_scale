Figure {
    id: figure
    Column {
        anchors.fill: parent
        Axis {
            handle: "ax"
            shareX: "axis"
            // LinePlot {}
            LinePlot {
                handle: "blue"
                line { color: "#589BDA"; style: "-"; width: 4 }
            }
            LinePlot {
                handle: "red"
                line { color: "#EF5762"; style: "--"; width: 4 }
            }
            LinePlot {
                handle: "green"
                line { color: "#8CCF5C"; style: ":"; width: 4 }
            }
        }
        Axis {
            handle: "ax"
            shareX: "axis"
            // LinePlot {}
            LinePlot {
                handle: "blue"
                line { color: "#589BDA"; style: "-"; width: 4 }
            }
            LinePlot {
                handle: "red"
                line { color: "#EF5762"; style: "--"; width: 4 }
            }
            LinePlot {
                handle: "green"
                line { color: "#8CCF5C"; style: ":"; width: 4 }
            }
        }
        Axis {
            handle: "ax"
            shareX: "axis"
            // LinePlot {}
            LinePlot {
                handle: "blue"
                line { color: "#589BDA"; style: "-"; width: 4 }
            }
            LinePlot {
                handle: "red"
                line { color: "#EF5762"; style: "--"; width: 4 }
            }
            LinePlot {
                handle: "green"
                line { color: "#8CCF5C"; style: ":"; width: 4 }
            }
        }
    }
}
