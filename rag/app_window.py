"""
Ventana Nativa de Escritorio para Tu Archivo (RAG).
Usa GTK3 + WebKit2GTK nativo del sistema para abrirse como una aplicación de escritorio independiente.
"""

from pathlib import Path

import gi

gi.require_version('Gtk', '3.0')
gi.require_version('Gdk', '3.0')
gi.require_version('WebKit2', '4.1')
from gi.repository import Gdk, GdkPixbuf, Gio, Gtk, WebKit2

URL = "http://localhost:7860"
ICON = Path.home() / ".local/share/icons/tu-archivo.png"

# Mismos colores de fondo que la interfaz web, para que no destelle en blanco al cargar.
BG_LIGHT = (251, 251, 253)
BG_DARK = (22, 22, 23)


def _rgba(rgb: tuple[int, int, int]) -> Gdk.RGBA:
    color = Gdk.RGBA()
    color.red, color.green, color.blue, color.alpha = (*(c / 255 for c in rgb), 1.0)
    return color


def _follow_system_theme(webview: WebKit2.WebView) -> None:
    """Sigue el modo claro/oscuro de GNOME. WebKitGTK deriva `prefers-color-scheme` del
    tema GTK, así que la interfaz web cambia junto con la ventana."""
    gtk_settings = Gtk.Settings.get_default()

    def apply(dark: bool) -> None:
        gtk_settings.set_property("gtk-application-prefer-dark-theme", dark)
        webview.set_background_color(_rgba(BG_DARK if dark else BG_LIGHT))

    try:
        schema = "org.gnome.desktop.interface"
        if Gio.SettingsSchemaSource.get_default().lookup(schema, True) is None:
            raise LookupError(schema)
        desktop = Gio.Settings.new(schema)
        is_dark = lambda: desktop.get_string("color-scheme") == "prefer-dark"
        desktop.connect("changed::color-scheme", lambda *_: apply(is_dark()))
        apply(is_dark())
    except Exception:
        apply(False)


def run_app():
    Gtk.init()
    win = Gtk.Window(title="Tu Archivo")
    win.set_default_size(1180, 820)
    win.set_size_request(420, 520)
    win.set_position(Gtk.WindowPosition.CENTER)
    win.set_wmclass("tu-archivo", "Tu Archivo")

    # Barra de título compacta del sistema, sin texto redundante.
    header = Gtk.HeaderBar(title="Tu Archivo", show_close_button=True)
    win.set_titlebar(header)

    try:
        win.set_icon(GdkPixbuf.Pixbuf.new_from_file(str(ICON)))
    except Exception:
        pass

    win.connect("destroy", Gtk.main_quit)

    webview = WebKit2.WebView()
    settings = webview.get_settings()
    settings.set_enable_javascript(True)
    settings.set_enable_smooth_scrolling(True)
    settings.set_enable_back_forward_navigation_gestures(False)
    settings.set_enable_write_console_messages_to_stdout(False)

    _follow_system_theme(webview)

    # Enlaces externos en el navegador del sistema, no dentro de la app.
    def on_decide_policy(_view, decision, decision_type):
        if decision_type == WebKit2.PolicyDecisionType.NAVIGATION_ACTION:
            uri = decision.get_navigation_action().get_request().get_uri()
            if not uri.startswith(URL):
                Gtk.show_uri_on_window(win, uri, Gdk.CURRENT_TIME)
                decision.ignore()
                return True
        return False

    webview.connect("decide-policy", on_decide_policy)

    webview.load_uri(URL)
    win.add(webview)
    win.show_all()
    Gtk.main()


if __name__ == "__main__":
    run_app()
