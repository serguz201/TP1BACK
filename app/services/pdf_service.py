"""Genera PDF de cotización con fpdf2."""

import unicodedata

from fpdf import FPDF

# Las fuentes "core" de fpdf2 (Helvetica y compañía) solo admiten latin-1.
#
# H-17. `multi_cell()` con el comentario del usuario reventaba con cualquier
# carácter fuera de ese juego —un emoji, el símbolo del euro— y devolvía un 500
# en la descarga del PDF. El campo de comentarios de la UI acepta 500 caracteres
# libres, así que llegar ahí es trivial: se reprodujo con "Carga urgente 🚢 —
# coste 100€". Incrustar una fuente TTF Unicode obligaría a distribuir el
# fichero de fuente con el backend; se opta por transliterar, que conserva el
# texto legible y no añade dependencias de despliegue.
_SUSTITUCIONES = {
    "€": "EUR", "—": "-", "–": "-", "“": '"', "”": '"',
    "‘": "'", "’": "'", "…": "...", "•": "-", "→": "->", "±": "+/-",
}


def _latin1(texto) -> str:
    """Devuelve `texto` representable en latin-1, sin perder legibilidad."""
    if texto is None:
        return ""
    s = str(texto)
    for orig, sust in _SUSTITUCIONES.items():
        s = s.replace(orig, sust)
    try:
        s.encode("latin-1")
        return s
    except UnicodeEncodeError:
        # Se descomponen los acentos y se descarta lo que no tenga equivalente
        # (emoji, ideogramas). Nunca lanza.
        normal = unicodedata.normalize("NFKD", s)
        return normal.encode("latin-1", "ignore").decode("latin-1")


def generate_quotation_pdf(quotation) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Cabecera ──────────────────────────────────────────────────────────────
    pdf.set_fill_color(11, 61, 92)  # azul corporativo
    pdf.rect(0, 0, 210, 40, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_y(10)
    pdf.cell(0, 10, "JPS Logistic S.A.C.", align="C", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, "Cotizacion de Flete Maritimo", align="C", ln=True)

    # ── Código y fecha ────────────────────────────────────────────────────────
    pdf.set_y(48)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(95, 8, _latin1(f"Cotizacion: {quotation.code}"), ln=False)
    fecha = quotation.created_at.strftime("%d/%m/%Y %H:%M")
    pdf.cell(95, 8, f"Fecha: {fecha}", align="R", ln=True)
    pdf.ln(2)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    # ── Detalles del embarque ─────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 8, "  Detalles del Embarque", fill=True, ln=True)
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 10)
    rows = [
        ("Puerto de Embarque:", quotation.puerto_origen),
        ("Puerto de Destino:", "Callao (PE)"),
        ("Peso Neto:", f"{quotation.peso_kg:,.0f} kg"),
    ]
    if quotation.importador:
        rows.insert(1, ("Importador:", quotation.importador))
    if quotation.tipo_contenedor:
        rows.append(("Tipo de Contenedor:", quotation.tipo_contenedor))
    if quotation.unidades:
        rows.append(("Unidades:", str(quotation.unidades)))
    if quotation.volumen_cbm:
        rows.append(("Volumen:", f"{quotation.volumen_cbm:.1f} CBM"))
    if quotation.fecha_embarque:
        rows.append(("Fecha de Embarque:", quotation.fecha_embarque))

    for label, value in rows:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(60, 7, _latin1(label))
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, _latin1(value), ln=True)

    pdf.ln(4)

    # ── Resultado de la predicción ────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 8, "  Estimacion de Flete", fill=True, ln=True)
    pdf.ln(4)

    # Monto principal
    pdf.set_fill_color(11, 61, 92)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 16, f"USD {quotation.flete_estimado_usd:,.2f}", align="C", fill=True, ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(
        0, 6,
        f"Intervalo de confianza 95%: USD {quotation.ic95_min:,.2f} - USD {quotation.ic95_max:,.2f}",
        align="C", ln=True,
    )
    # H-18: el PDF omitia `mape_regimen` y `mape_modelo`, pese a que la
    # migracion 004 se anadio precisamente para que "una cotizacion guardada o
    # SU PDF" pudieran decir si la estimacion se apoyo en mercado observado o
    # congelado. El cliente recibia un intervalo sin saber que el error esperado
    # de esa cotizacion era ~27.5% y no el 22.2% de referencia.
    _extrapolada = (quotation.mape_regimen or "").lower() == "extrapolado"
    if quotation.mape_regimen:
        pdf.cell(
            0, 5,
            _latin1(
                f"Regimen: {'mercado proyectado' if _extrapolada else 'mercado observado'}"
                f"  ·  error esperado del modelo: +/-{quotation.mape_modelo:.1f}%"
            ),
            align="C", ln=True,
        )
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    if _extrapolada:
        pdf.set_fill_color(255, 247, 224)
        pdf.set_draw_color(245, 200, 120)
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(
            0, 5,
            _latin1(
                "AVISO: la fecha de embarque solicitada cae fuera del periodo de "
                "mercado que el modelo observo al entrenarse. La estimacion se "
                "calcula con el ultimo mercado conocido, y su error esperado "
                f"(+/-{quotation.mape_modelo:.1f}%) es mayor que el medido dentro "
                "del historico. Uselo como referencia orientativa."
            ),
            border=1, fill=True, align="L",
        )
        pdf.ln(3)

    # ── Comentario ────────────────────────────────────────────────────────────
    if quotation.comentario:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(0, 8, "  Comentarios", fill=True, ln=True)
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, _latin1(quotation.comentario))
        pdf.ln(4)

    # ── Pie de página ─────────────────────────────────────────────────────────
    pdf.set_y(-25)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(130, 130, 130)
    pdf.cell(0, 5, "Documento generado automaticamente por JPS Freight Predictor.", align="C", ln=True)
    pdf.cell(0, 5, "Esta cotizacion es referencial y puede variar segun condiciones del mercado.", align="C", ln=True)

    return bytes(pdf.output())
