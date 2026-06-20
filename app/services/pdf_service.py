"""Genera PDF de cotización con fpdf2."""

from io import BytesIO
from datetime import datetime, timezone

from fpdf import FPDF


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
    pdf.cell(95, 8, f"Cotizacion: {quotation.code}", ln=False)
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
        ("Puerto de Origen:", quotation.puerto_origen),
        ("Puerto de Destino:", "Callao (PE)"),
        ("Tipo de Contenedor:", quotation.tipo_contenedor),
        ("Peso Neto:", f"{quotation.peso_kg:,.0f} kg"),
    ]
    if quotation.unidades:
        rows.append(("Unidades:", str(quotation.unidades)))
    if quotation.volumen_cbm:
        rows.append(("Volumen:", f"{quotation.volumen_cbm:.1f} CBM"))
    if quotation.fecha_embarque:
        rows.append(("Fecha de Embarque:", quotation.fecha_embarque))

    for label, value in rows:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(60, 7, label)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, value, ln=True)

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
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    # ── Comentario ────────────────────────────────────────────────────────────
    if quotation.comentario:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(0, 8, "  Comentarios", fill=True, ln=True)
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, quotation.comentario)
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
