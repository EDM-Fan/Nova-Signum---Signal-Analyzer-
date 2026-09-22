import sys
from PySide6.QtWidgets import QApplication, QMainWindow

app = QApplication(sys.argv)
win = QMainWindow()
win.setStyleSheet("background-color: red;")
win.resize(400, 300)
win.show()
sys.exit(app.exec())
