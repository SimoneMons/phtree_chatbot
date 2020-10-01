from PyQt5 import QtCore, QtGui, QtWidgets


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.output_te = QtWidgets.QTextEdit(readOnly=True)
        self.input_le = QtWidgets.QLineEdit(returnPressed=self.on_return_pressed)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        lay = QtWidgets.QVBoxLayout(central_widget)
        lay.addWidget(self.output_te)
        lay.addWidget(self.input_le)

        self.output_te.setPlainText(
            "hello your welcome ask anything.Type 'quit' in lower case for leave"
        )


    @QtCore.pyqtSlot()
    def on_return_pressed(self):
        text = self.input_le.text()
        if text:
            # response
            res = 'Holaaaaaaaaaaa'
            self.output_te.append("[me]: {}".format(text))
            self.output_te.append("[bot]: {}".format(res))
            self.input_le.clear()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())