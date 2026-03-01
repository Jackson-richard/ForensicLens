import io
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_forensic_report(
    case_id: str,
    file_name: str,
    file_hash: str,
    authenticity_score: float,
    generator_attribution: str,
    metadata: dict,
    heatmap_bytes: bytes = None
) -> bytes:
    """Generates a structured PDF report using ReportLab."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    title_style.alignment = 1 
    
    normal_style = styles['Normal']
    
    elements = []
    
   
    elements.append(Paragraph("Forensic Analysis Report", title_style))
    elements.append(Spacer(1, 12))
    
    
    data = [
        ["Report Timestamp:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["Case ID:", case_id],
        ["File Name:", file_name],
        ["SHA-256 Hash:", file_hash],
    ]

    t_case = Table(data, colWidths=[120, 300])
    t_case.setStyle(TableStyle([
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    elements.append(t_case)
    elements.append(Spacer(1, 24))
    
    
    elements.append(Paragraph("Analysis Results", styles['Heading2']))
    is_fake = authenticity_score > 0.5
    verdict = "LIKELY MANIPULATED" if is_fake else "LIKELY AUTHENTIC"
    verdict_color = colors.red if is_fake else colors.green

    results_data = [
        ["Verdict:", Paragraph(f"<font color='{verdict_color.hexval()}'><b>{verdict}</b></font>", normal_style)],
        ["Manipulation Score:", f"{authenticity_score:.2%} (Confidence)"],
        ["Generator Attribution:", generator_attribution]
    ]
    t_results = Table(results_data, colWidths=[140, 280])
    t_results.setStyle(TableStyle([
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    elements.append(t_results)
    elements.append(Spacer(1, 24))
    
   
    if heatmap_bytes:
        elements.append(Paragraph("Explainability Map (Grad-CAM):", styles['Heading3']))
        try:
            img_buffer = io.BytesIO(heatmap_bytes)
            img = Image(img_buffer, width=300, height=300)
            img.hAlign = 'CENTER'
            elements.append(img)
        except Exception as e:
            elements.append(Paragraph(f"Heatmap rendering failed: {e}", normal_style))
        elements.append(Spacer(1, 24))
        
   
    if metadata:
        elements.append(Paragraph("Extracted Metadata:", styles['Heading3']))
        
        
        meta_data_list = []
        for k, v in metadata.items():
            key_str = str(k)[:40]
            val_str = str(v)[:80] + ("..." if len(str(v)) > 80 else "")
            meta_data_list.append([key_str, val_str])
        
        if meta_data_list:
            t_meta = Table(meta_data_list, colWidths=[150, 270])
            t_meta.setStyle(TableStyle([
                ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ]))
            elements.append(t_meta)
    
    doc.build(elements)
    
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
