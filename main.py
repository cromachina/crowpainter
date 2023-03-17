import sys

from PySide6.QtCore import *
from PySide6.QtWidgets import *
from PySide6.QtGui import *

from pyrsistent import *

class CrowPainter(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CrowPainter")

if __name__ == "__main__":
    app = QApplication(sys.argv)

    widget = CrowPainter()
    widget.show()

    sys.exit(app.exec_())