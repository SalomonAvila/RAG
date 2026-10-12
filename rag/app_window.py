"""
Ventana Nativa de Escritorio para Tu Archivo (RAG).
Usa GTK3 + WebKit2GTK nativo del sistema para abrirse como una aplicación de escritorio independiente.
"""

import sys
import gi

gi.require_version('Gtk', '3.0')
gi.require_version('WebKit2', '4.1')
from gi.repository import Gtk, WebKit2, GdkPixbuf

def run_app():
    Gtk.init()
    win = Gtk.Window(title="Tu Archivo · RAG Homelab")
    win.set_default_size(1260, 860)
    win.set_position(Gtk.WindowPosition.CENTER)

    try:
        pixbuf = GdkPixbuf.Pixbuf.new_from_file("/home/micho/.local/share/icons/tu-archivo.png")
        win.set_icon(pixbuf)
    except Exception:
        pass

    win.connect("destroy", Gtk.main_quit)

    webview = WebKit2.WebView()
    # Habilitar soporte moderno de HTML5 y almacenamiento
    settings = webview.get_settings()
    settings.set_enable_javascript(True)
    settings.set_enable_write_console_messages_to_stdout(False)

    webview.load_uri("http://localhost:7860")
    win.add(webview)
    win.show_all()
    Gtk.main()

if __name__ == "__main__":
    run_app()
