Figure {
    id: figure
    Layouts.Column {
        spacing: 0
        Layouts.Repeater {
            count: 2
            Axis {
                handle: "ax"
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
                    line { color: "#8CCF5C"; style: "--"; width: 2 }
                }

                LinePlot {
                    handle: "greenDots"
                    line { color: "#8CCF5C"; style: "."; width: 5 }
                }

                CanvasPlot {
                    id: useful
                    handle: "useful"

                    property real min: 0
                    property real max: 0
                    property color color: "#AAFFFFFF"

                    property var data: []
                    onDataChanged: updatePlot()

                    Component.onCompleted: {
                        registerProperties({
                            "data": "data",
                            "min": "min",
                            "max": "max",
                            "color": "color"
                        })
                    }

                    property var blocks: []

                    function updatePlot() {
                        while (blocks.length < data.length)
                            blocks.push(blockComp.createObject(useful))

                        while (blocks.length > data.length)
                            blocks.pop().destroy()

                        var i
                        for (i = 0; i < data.length; ++i) {
                            var edges = data[i]
                            var block = blocks[i]
                            block.x = edges[0]
                            block.width = edges[1] - edges[0]
                        }
                    }

                    Component {
                        id: blockComp

                        CanvasRect {
                            id: rect
                            fill.color: useful.color
                            line.color: "#00FFFFFF"
                            y: useful.min
                            height: useful.max - useful.min
                        }
                    }
                }
            }
        }
    }
}
