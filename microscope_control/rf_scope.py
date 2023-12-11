from PyQt5.QtWidgets import QApplication

from daq import Daq
from daq_window import StartWindow
# launch remote focusing control GUI

card = Daq('Dev1/', 50000)

app = QApplication([])
start_window = StartWindow(card)
start_window.show()

app.exit(app.exec_())