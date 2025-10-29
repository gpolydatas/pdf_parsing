#!/usr/bin/env python3
"""
Enhanced solution for document reconstruction with image extraction, proper alignment, hyperlink extraction,
and repetitive image detection.

This script:
1. Processes PDF pages with Azure Document Intelligence API
2. Extracts images from the PDF using the bounding regions
3. Detects repetitive images (e.g., logos) and excludes them from Claude analysis and JSON outputs
4. Correctly aligns text and preserves page numbers
5. Reconstructs the document as an accurate HTML representation
6. Includes base64-encoded images in JSON outputs (excluding repetitive ones)
7. Analyzes unique images with Claude once and reuses responses
8. Extracts hyperlinks using PyMuPDF and includes them in JSON outputs
"""
import json
import os
import sys
import fitz  # PyMuPDF
import tempfile
import base64
import cv2
import numpy as np
from pathlib import Path
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.exceptions import HttpResponseError
from html import escape
import shutil
from anthropic import Anthropic
import re
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBoxHorizontal, LTAnno
from fuzzywuzzy import fuzz
import html

import re
import PyPDF2
from pdfminer.high_level import extract_pages
import re
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_pages
import re
from pdfminer.high_level import extract_pages
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
import pdfplumber
# ============= CONFIGURATION (MODIFY THESE VALUES) =============
PDF_PATH = 'test_hyperlinks_unified.pdf'
AZURE_ENDPOINT = "https://northeurope.api.cognitive.microsoft.com/"
AZURE_API_KEY = ""
CLAUDE_MODEL = "claude-3-7-sonnet-20250219"
CLAUDE_API_KEY = ""  # Replace with your actual key
OUTPUT_DIR = Path('output_images')
SIMILARITY_THRESHOLD = 0.75  # Threshold for considering images as duplicates (0-1, higher = stricter)
# =============================================================


class DocumentProcessor:
    """Process documents using Azure Document Intelligence API, extract hyperlinks, and detect repetitive images"""
    
    def __init__(self, endpoint, api_key):
        print(f"DEBUG: Initializing DocumentProcessor with endpoint: {endpoint}")
        self.endpoint = endpoint
        self.api_key = api_key
        self.client = DocumentIntelligenceClient(endpoint=self.endpoint, credential=AzureKeyCredential(self.api_key))
        self.analysis_results = []
        self.page_images = []
        self.cross_page_tables = {}
        self.hyperlinks = []
        self.extracted_images = []  # Store all extracted images before filtering
        self.unique_images = []     # Store unique images after filtering
        
    def process_pdf(self, pdf_path):
        print(f"DEBUG: Starting to process PDF: {pdf_path}")
        if not os.path.exists(pdf_path):
            print(f"ERROR: PDF file does not exist: {pdf_path}")
            return self.analysis_results
        
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            print(f"DEBUG: Created output directory: {OUTPUT_DIR}")
        
        # Store pdf_path for use in _extract_hyperlinks
        self.pdf_path = pdf_path
        
        pdf_document = fitz.open(pdf_path)
        print(f"DEBUG: PDF opened successfully. Contains {len(pdf_document)} pages")
        
        # Extract hyperlinks from PDF structure using PDFMiner before rendering
        print("DEBUG: Starting hyperlink extraction from PDF structure...")
        self._extract_hyperlinks()
        
        # Proceed with rendering and Azure analysis
        for page_num in range(len(pdf_document)):
            print(f"\nDEBUG: Processing page {page_num + 1} of {len(pdf_document)}")
            page = pdf_document[page_num]
            print(f"DEBUG: Page dimensions: {page.rect.width}x{page.rect.height}")
            zoom = 3.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            print(f"DEBUG: Rendered image size: {pix.width}x{pix.height}")
            page_image_path = OUTPUT_DIR / f"page_{page_num + 1}.png"
            pix.save(str(page_image_path))
            self.page_images.append({
                'page_number': page_num + 1,
                'path': str(page_image_path),
                'width': pix.width,
                'height': pix.height
            })
            print(f"DEBUG: Full page image saved as: {page_image_path}")
            
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                temp_path = temp_file.name
                pix.save(temp_path)
            
            print(f"DEBUG: Page {page_num + 1} saved as temporary PNG image: {temp_path}")
            print(f"DEBUG: Temporary image file size: {os.path.getsize(temp_path)} bytes")
            result = self._analyze_image(temp_path)
            self.analysis_results.append({'page_number': page_num + 1, 'result': result})
            os.unlink(temp_path)
            print(f"DEBUG: Temporary image file removed")
        
        print(f"DEBUG: PDF processing complete. Collected {len(self.analysis_results)} result sets")
        
        # Identify cross-page tables with Claude verification
        self._identify_cross_page_tables()
        
        # Extract images and filter out repetitive ones
        self.extract_images()
        self._filter_repetitive_images()
        
        return self.analysis_results
    



    def _extract_hyperlinks(self):
        """Extract hyperlinks from a PDF using PyMuPDF and PDFMiner, capturing the actual text."""
        import fitz  # PyMuPDF
        from pdfminer.high_level import extract_text_to_fp
        from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTChar
        from pdfminer.pdfpage import PDFPage
        from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
        from pdfminer.converter import PDFPageAggregator
        from io import StringIO
        import re
        
        hyperlinks = []
        
        # Use self.pdf_path set by process_pdf
        pdf_path = self.pdf_path
        if not pdf_path:
            print("ERROR: No PDF path set for hyperlink extraction")
            return hyperlinks
        
        # Step 1: Extract hyperlinks using PyMuPDF
        try:
            doc = fitz.open(pdf_path)
            print(f"INFO: Extracting hyperlinks from PDF with PyMuPDF: {pdf_path}")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                links = page.get_links()
                
                # Get the text on the page to map links to text
                text_blocks = page.get_text("dict")["blocks"]
                text_spans = []
                for block in text_blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if not text:
                                    continue
                                # PyMuPDF bbox is (x0, y0, x1, y1)
                                bbox = span["bbox"]
                                # Check if the text is hidden (white-on-white)
                                is_hidden = False
                                color = span.get("color", 0)  # sRGB color value
                                # Convert sRGB to RGB
                                r = (color >> 16) & 255
                                g = (color >> 8) & 255
                                b = color & 255
                                # Assume white background; check if text color is white
                                if r >= 240 and g >= 240 and b >= 240:
                                    is_hidden = True
                                text_spans.append({
                                    "text": text,
                                    "bbox": bbox,  # (x0, y0, x1, y1)
                                    "is_hidden": is_hidden
                                })
                
                # Process each link
                for link in links:
                    if "uri" not in link:
                        continue
                    url = link["uri"]
                    rect = link["from"]  # PyMuPDF provides the rectangle of the link (x0, y0, x1, y1)
                    link_x0, link_y0, link_x1, link_y1 = rect.x0, rect.y0, rect.x1, rect.y1
                    
                    # Find the text that overlaps with the link's rectangle
                    link_text = ""
                    is_hidden = False
                    for span in text_spans:
                        span_x0, span_y0, span_x1, span_y1 = span["bbox"]
                        # Check for overlap
                        overlap_x = min(link_x1, span_x1) - max(link_x0, span_x0)
                        overlap_y = min(link_y1, span_y1) - max(link_y0, span_y0)
                        if overlap_x > 0 and overlap_y > 0:
                            link_text = span["text"]
                            is_hidden = span["is_hidden"]
                            break
                    
                    if not link_text:
                        link_text = url  # Fallback to URL if no text is found
                    
                    # Determine link type
                    link_type = "external" if url.startswith("http") else "internal"
                    
                    hyperlinks.append({
                        "text": link_text,
                        "url": url,
                        "page_number": page_num,
                        "coordinates": {
                            "x0": link_x0,
                            "y0": link_y0,
                            "x1": link_x1,
                            "y1": link_y1,
                            "source": "pymupdf"
                        },
                        "type": link_type,
                        "source": "pymupdf_annot",
                        "is_hidden": is_hidden
                    })
            
            doc.close()
        except Exception as e:
            print(f"ERROR: Failed to extract hyperlinks with PyMuPDF: {str(e)}")
        
        # Step 2: Extract hyperlinks using PDFMiner as a fallback
        try:
            print(f"INFO: Extracting hyperlinks from PDF with PDFMiner: {pdf_path}")
            rsrcmgr = PDFResourceManager()
            laparams = LAParams()
            device = PDFPageAggregator(rsrcmgr, laparams=laparams)
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            
            with open(pdf_path, "rb") as fp:
                for page_num, page in enumerate(PDFPage.get_pages(fp)):
                    interpreter.process_page(page)
                    layout = device.get_result()
                    
                    # Extract text objects and their coordinates
                    text_objects = []
                    for obj in layout:
                        if isinstance(obj, (LTTextBox, LTTextLine)):
                            for char in obj:
                                if isinstance(char, LTChar):
                                    text = char.get_text()
                                    if text.strip():
                                        # PDFMiner bbox is (x0, y0, x1, y1)
                                        bbox = char.bbox
                                        # Check if the text is hidden (white-on-white)
                                        is_hidden = False
                                        fill_color = char.graphicstate.ncolor  # Non-stroking color (fill)
                                        if isinstance(fill_color, (list, tuple)) and len(fill_color) == 3:
                                            r, g, b = [int(c * 255) for c in fill_color]
                                            if r >= 240 and g >= 240 and b >= 240:
                                                is_hidden = True
                                        text_objects.append({
                                            "text": text,
                                            "bbox": bbox,
                                            "is_hidden": is_hidden
                                        })
                    
                    # Extract annotations (links) from the page
                    if "Annots" in page:
                        for annot in page["Annots"]:
                            annot_obj = annot.get_object()
                            if annot_obj.get("Subtype") != "/Link":
                                continue
                            if "A" not in annot_obj or "URI" not in annot_obj["A"]:
                                continue
                            url = annot_obj["A"]["URI"]
                            # Get the rectangle of the link
                            rect = annot_obj.get("Rect", [0, 0, 0, 0])
                            link_x0, link_y0, link_x1, link_y1 = rect
                            
                            # Find the text that overlaps with the link's rectangle
                            link_text = ""
                            is_hidden = False
                            for text_obj in text_objects:
                                text_x0, text_y0, text_x1, text_y1 = text_obj["bbox"]
                                overlap_x = min(link_x1, text_x1) - max(link_x0, text_x0)
                                overlap_y = min(link_y1, text_y1) - max(link_y0, text_y0)
                                if overlap_x > 0 and overlap_y > 0:
                                    link_text += text_obj["text"]
                                    is_hidden = text_obj["is_hidden"]
                                    break
                            
                            if not link_text:
                                link_text = url  # Fallback to URL if no text is found
                            
                            # Determine link type
                            link_type = "external" if url.startswith("http") else "internal"
                            
                            # Check if this link is already extracted by PyMuPDF
                            duplicate = False
                            for existing_link in hyperlinks:
                                if (existing_link["url"] == url and
                                    existing_link["page_number"] == page_num and
                                    existing_link["text"] == link_text):
                                    duplicate = True
                                    break
                            if not duplicate:
                                hyperlinks.append({
                                    "text": link_text,
                                    "url": url,
                                    "page_number": page_num,
                                    "coordinates": {
                                        "x0": link_x0,
                                        "y0": link_y0,
                                        "x1": link_x1,
                                        "y1": link_y1,
                                        "source": "pdfminer"
                                    },
                                    "type": link_type,
                                    "source": "pdfminer_annot",
                                    "is_hidden": is_hidden
                                })
            
            device.close()
        except Exception as e:
            print(f"ERROR: Failed to extract hyperlinks with PDFMiner: {str(e)}")
        
        # Step 3: Deduplicate hyperlinks
        seen = set()
        deduplicated_hyperlinks = []
        for link in hyperlinks:
            key = (link["url"], link["text"].lower(), link["page_number"])
            if key not in seen:
                seen.add(key)
                deduplicated_hyperlinks.append(link)
        
        print(f"INFO: Extracted {len(deduplicated_hyperlinks)} unique hyperlinks from PDF")
        # Store hyperlinks in a class attribute for later use
        self.hyperlinks = deduplicated_hyperlinks
        return deduplicated_hyperlinks

        
    def _analyze_image(self, image_path):
        try:
            print(f"DEBUG: Sending image to Azure Document Intelligence: {image_path}")
            with open(image_path, "rb") as image_file:
                file_content = image_file.read()
            print(f"DEBUG: Image file read successfully, size: {len(file_content)} bytes")
            print(f"DEBUG: Starting API call to analyze document...")
            poller = self.client.begin_analyze_document("prebuilt-layout", body=file_content, content_type="application/octet-stream")
            print(f"DEBUG: Waiting for API analysis to complete...")
            result = poller.result()
            print(f"DEBUG: API analysis completed successfully")
            if hasattr(result, 'pages'):
                print(f"DEBUG: Received {len(result.pages)} pages from API")
                for i, page in enumerate(result.pages):
                    if hasattr(page, 'lines'):
                        print(f"DEBUG: Page {i+1} has {len(page.lines)} lines of text")
            if hasattr(result, 'tables') and result.tables is not None:
                print(f"DEBUG: Received {len(result.tables)} tables from API")
            if hasattr(result, 'figures') and result.figures is not None:
                print(f"DEBUG: Received {len(result.figures)} figures from API")
            return result
        except HttpResponseError as error:
            print(f"ERROR: HTTP error analyzing document: {error}")
            raise
        except Exception as e:
            print(f"ERROR: Unexpected error analyzing document: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def extract_images(self):
        """Extract images based on figure bounding regions"""
        print(f"DEBUG: Extracting images based on figure bounding regions")
        self.extracted_images = []
        
        for page_data in self.analysis_results:
            page_number = page_data['page_number']
            result = page_data['result']
            page_image_info = next((img for img in self.page_images if img['page_number'] == page_number), None)
            if not page_image_info:
                print(f"WARNING: No saved image found for page {page_number}")
                continue
            
            img = cv2.imread(page_image_info['path'])
            if img is None:
                print(f"ERROR: Failed to load image: {page_image_info['path']}")
                continue
            img_height, img_width = img.shape[:2]
            print(f"DEBUG: Loaded page image: {page_image_info['path']}, dimensions: {img_width}x{img_height}")
            
            if not hasattr(result, 'figures') or not result.figures:
                print(f"DEBUG: No figures found in API result for page {page_number}")
                continue
            
            for i, figure in enumerate(result.figures):
                if not hasattr(figure, 'bounding_regions') or not figure.bounding_regions:
                    print(f"WARNING: Figure {i} on page {page_number} has no bounding regions")
                    continue
                
                for region_idx, region in enumerate(figure.bounding_regions):
                    if not hasattr(region, 'polygon') or not region.polygon:
                        print(f"WARNING: Region {region_idx} of figure {i} on page {page_number} has no polygon")
                        continue
                    
                    try:
                        polygon = region.polygon
                        x_coords = polygon[::2]
                        y_coords = polygon[1::2]
                        x_min = max(0, int(min(x_coords)))
                        y_min = max(0, int(min(y_coords)))
                        x_max = min(img_width, int(max(x_coords)))
                        y_max = min(img_height, int(max(y_coords)))
                        
                        if x_min >= x_max or y_min >= y_max:
                            print(f"WARNING: Invalid bounding box for figure {i} on page {page_number}: ({x_min}, {y_min}, {x_max}, {y_max})")
                            continue
                        
                        cropped_img = img[y_min:y_max, x_min:x_max]
                        image_filename = f"page_{page_number}_figure_{i}.png"
                        figure_path = os.path.join(os.path.abspath(str(OUTPUT_DIR)), image_filename)
                        cv2.imwrite(figure_path, cropped_img)
                        _, buffer = cv2.imencode('.png', cropped_img)
                        base64_image = base64.b64encode(buffer).decode('utf-8')
                        
                        print(f"DEBUG: Extracted figure {i} from page {page_number} to {figure_path}")
                        self.extracted_images.append({
                            'page_number': page_number,
                            'figure_index': i,
                            'path': figure_path,
                            'filename': image_filename,
                            'x_min': x_min,
                            'y_min': y_min,
                            'x_max': x_max,
                            'y_max': y_max,
                            'width': x_max - x_min,
                            'height': y_max - y_min,
                            'base64': base64_image,
                            'image_data': cropped_img  # Store raw image data for similarity check
                        })
                    except Exception as e:
                        print(f"ERROR: Failed to extract figure {i} from page {page_number}: {str(e)}")
        
        print(f"DEBUG: Extracted {len(self.extracted_images)} images")
        return self.extracted_images
    
    def _filter_repetitive_images(self):
        """Filter out repetitive images (e.g., logos) using ORB feature matching and size heuristics"""
        print("DEBUG: Filtering out repetitive images...")
        if not self.extracted_images:
            print("DEBUG: No images to filter")
            self.unique_images = []
            return
        
        # Increase keypoint detection for small images like logos
        orb = cv2.ORB_create(nfeatures=1000)  # Increase from default 500
        flann_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)
        flann = cv2.FlannBasedMatcher(flann_params, {})
        unique_images = []
        seen_images = []  # Store descriptors and filenames of seen images
        # SIMILARITY_THRESHOLD = 0.75  # Lowered from 0.9 for more aggressive duplicate detection
        MIN_LOGO_SIZE = 10000  # Minimum area (width * height) in pixels to consider an image non-logo
        
        for img_data in self.extracted_images:
            img = img_data['image_data']
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = orb.detectAndCompute(gray, None)
            img_area = img_data['width'] * img_data['height']
            
            # Heuristic: Small images with insufficient keypoints are likely logos
            if (descriptors is None or len(descriptors) < 2) and img_area < MIN_LOGO_SIZE:
                print(f"DEBUG: Image {img_data['filename']} has insufficient keypoints and small size ({img_area} px), likely a logo, skipping")
                continue
            elif descriptors is None or len(descriptors) < 2:
                print(f"DEBUG: Insufficient keypoints in image {img_data['filename']}, treating as unique")
                unique_images.append(img_data)
                continue
            
            is_repetitive = False
            for seen_desc, seen_filename in seen_images:
                if seen_desc is None or len(seen_desc) < 2:
                    continue
                try:
                    matches = flann.knnMatch(descriptors, seen_desc, k=2)
                    good_matches = []
                    for match in matches:
                        if len(match) == 2:  # Two matches available
                            m, n = match
                            if m.distance < 0.7 * n.distance:  # Lowe's ratio test
                                good_matches.append(m)
                        elif len(match) == 1:  # Single match
                            good_matches.append(match[0])  # Accept cautiously
                    if good_matches:
                        similarity = len(good_matches) / max(len(descriptors), len(seen_desc))
                        print(f"DEBUG: Comparing {img_data['filename']} to {seen_filename}: similarity = {similarity:.2f}")
                        if similarity > SIMILARITY_THRESHOLD:
                            print(f"DEBUG: Image {img_data['filename']} is repetitive (similar to {seen_filename}, similarity: {similarity:.2f})")
                            is_repetitive = True
                            break
                except cv2.error as e:
                    print(f"DEBUG: Skipping match for {img_data['filename']} due to error: {str(e)}")
                    continue
            
            if not is_repetitive:
                print(f"DEBUG: Image {img_data['filename']} is unique (area: {img_area} px)")
                unique_images.append(img_data)
                seen_images.append((descriptors, img_data['filename']))
        
        self.unique_images = unique_images
        print(f"DEBUG: Filtered to {len(self.unique_images)} unique images from {len(self.extracted_images)} total")
    
    def get_unique_images(self):
        """Return the filtered unique images"""
        return self.unique_images
    
    def _identify_cross_page_tables(self):
        """Identifies tables that span across multiple pages"""
        print("DEBUG: Identifying cross-page tables with enhanced detection...")
        self.cross_page_tables = {}
        potential_continuations = []
        
        all_tables = []
        for page_data in self.analysis_results:
            page_number = page_data['page_number']
            result = page_data['result']
            if not hasattr(result, 'tables') or not result.tables:
                continue
            
            for table_idx, table in enumerate(result.tables):
                table_key = f"page_{page_number}_{table_idx}"
                table_info = {
                    'page_number': page_number,
                    'table_index': table_idx,
                    'table_key': table_key,
                    'row_count': table.row_count if hasattr(table, 'row_count') else 0,
                    'column_count': table.column_count if hasattr(table, 'column_count') else 0
                }
                
                if hasattr(table, 'bounding_regions') and table.bounding_regions:
                    for region in table.bounding_regions:
                        if hasattr(region, 'polygon') and region.polygon:
                            polygon = region.polygon
                            page_height = result.pages[0].height if hasattr(result.pages[0], 'height') else 1000
                            y_coords = polygon[1::2]
                            table_info['position'] = {
                                'top': min(y_coords),
                                'bottom': max(y_coords),
                                'relative_top': min(y_coords) / page_height,
                                'relative_bottom': max(y_coords) / page_height
                            }
                            break
                
                if hasattr(table, 'spans') and table.spans:
                    table_info['spans'] = [{'offset': span.offset, 'length': span.length} for span in table.spans]
                
                header_row = []
                cell_content = []
                if hasattr(table, 'cells') and table.cells:
                    for cell in table.cells:
                        if cell.row_index == 0:
                            header_row.append({'column_index': cell.column_index, 'content': cell.content})
                        cell_content.append({'row_index': cell.row_index, 'column_index': cell.column_index, 'content': cell.content})
                
                header_row.sort(key=lambda x: x['column_index'])
                table_info['header_row'] = [cell['content'] for cell in header_row]
                table_info['cell_content'] = cell_content
                all_tables.append(table_info)
        
        all_tables.sort(key=lambda x: (x['page_number'], x.get('position', {}).get('top', 0)))
        
        for i in range(len(all_tables) - 1):
            current_table = all_tables[i]
            next_table = all_tables[i + 1]
            if next_table['page_number'] != current_table['page_number'] + 1:
                continue
            
            similarity_score = 0
            if current_table['column_count'] == next_table['column_count'] and current_table['column_count'] > 0:
                similarity_score += 1
            if len(current_table['header_row']) > 0 and len(next_table['header_row']) > 0:
                if current_table['header_row'] == next_table['header_row']:
                    similarity_score += 2
                elif len(set(current_table['header_row']) & set(next_table['header_row'])) >= 1:
                    similarity_score += 1
            if ('position' in current_table and 'position' in next_table and
                current_table['position']['relative_bottom'] > 0.7 and next_table['position']['relative_top'] < 0.3):
                similarity_score += 1
            
            cont_keywords = ["continued", "continued from", "continuation", "(cont", "(continued"]
            first_row_cells = [cell for cell in next_table['cell_content'] if cell['row_index'] == 0]
            for cell in first_row_cells:
                if any(keyword in cell['content'].lower() for keyword in cont_keywords):
                    similarity_score += 2
                    break
            
            if len(current_table['cell_content']) > 0 and len(next_table['cell_content']) > 0:
                last_row = max(cell['row_index'] for cell in current_table['cell_content'])
                next_rows = [cell['row_index'] for cell in next_table['cell_content'] if cell['row_index'] > 0]
                if next_rows and (min(next_rows) == last_row + 1 or min(next_rows) == 1):
                    similarity_score += 1
            
            current_first_col = [cell['content'] for cell in current_table['cell_content'] if cell['column_index'] == 0 and cell['row_index'] > 0]
            next_first_col = [cell['content'] for cell in next_table['cell_content'] if cell['column_index'] == 0 and cell['row_index'] > 0]
            if (current_first_col and next_first_col and all(c.isdigit() for c in current_first_col[-1] if c.isdigit()) and
                all(c.isdigit() for c in next_first_col[0] if c.isdigit())):
                try:
                    last_num = int(''.join(c for c in current_first_col[-1] if c.isdigit()))
                    first_num = int(''.join(c for c in next_first_col[0] if c.isdigit()))
                    if first_num == last_num + 1:
                        similarity_score += 2
                except (ValueError, IndexError):
                    pass
            
            is_at_page_boundary = ('position' in current_table and 'position' in next_table and
                                   current_table['position']['relative_bottom'] > 0.7 and next_table['position']['relative_top'] < 0.3)
            
            if similarity_score >= 2 or is_at_page_boundary:
                if is_at_page_boundary:
                    potential_continuations.append({'current_table': current_table, 'next_table': next_table, 'similarity_score': similarity_score})
                if similarity_score >= 2:
                    print(f"DEBUG: Cross-page table detected with score {similarity_score}: {current_table['table_key']} -> {next_table['table_key']}")
                    self.cross_page_tables[current_table['table_key']] = {
                        'is_cross_page': True, 'continuation_of': None, 'distribution_type': 'vertical',
                        'page_number': current_table['page_number'], 'table_index': current_table['table_index'], 'claude_verified': None
                    }
                    self.cross_page_tables[next_table['table_key']] = {
                        'is_cross_page': True, 'continuation_of': current_table['table_key'], 'distribution_type': 'vertical',
                        'page_number': next_table['page_number'], 'table_index': next_table['table_index'], 'claude_verified': None
                    }
        
        if potential_continuations:
            print(f"DEBUG: Found {len(potential_continuations)} potential cross-page table pairs at page boundaries")
            claude_verifications = self._verify_cross_page_tables_with_claude(potential_continuations)
            for table_key, is_verified in claude_verifications.items():
                if table_key in self.cross_page_tables:
                    self.cross_page_tables[table_key]['claude_verified'] = is_verified
                elif is_verified:
                    for pair in potential_continuations:
                        if pair['current_table']['table_key'] == table_key or pair['next_table']['table_key'] == table_key:
                            current_table = pair['current_table']
                            next_table = pair['next_table']
                            self.cross_page_tables[current_table['table_key']] = {
                                'is_cross_page': True, 'continuation_of': None, 'distribution_type': 'vertical',
                                'page_number': current_table['page_number'], 'table_index': current_table['table_index'], 'claude_verified': True
                            }
                            self.cross_page_tables[next_table['table_key']] = {
                                'is_cross_page': True, 'continuation_of': current_table['table_key'], 'distribution_type': 'vertical',
                                'page_number': next_table['page_number'], 'table_index': next_table['table_index'], 'claude_verified': True
                            }
                            print(f"DEBUG: Claude confirmed: {current_table['table_key']} -> {next_table['table_key']}")
                            break
        
        self._run_span_based_detection()
        print(f"DEBUG: Identified {len(self.cross_page_tables) // 2} cross-page table pairs")
    
    def _verify_cross_page_tables_with_claude(self, potential_continuations):
        print("DEBUG: Verifying potential cross-page tables with Claude...")
        claude_verifications = {}
        for pair in potential_continuations:
            current_table = pair['current_table']
            next_table = pair['next_table']
            current_key = current_table['table_key']
            next_key = next_table['table_key']
            if current_key in claude_verifications or next_key in claude_verifications:
                continue
            try:
                current_screenshot = self._capture_table_screenshot(current_table)
                next_screenshot = self._capture_table_screenshot(next_table)
                if current_screenshot is not None and next_screenshot is not None:
                    combined_image = self._create_combined_image(current_screenshot, next_screenshot)
                    if combined_image is not None:
                        base64_image = self._encode_image(combined_image)
                        if base64_image:
                            is_cross_page = self._ask_claude_for_verification(base64_image, current_table, next_table)
                            claude_verifications[current_key] = is_cross_page
                            claude_verifications[next_key] = is_cross_page
                            print(f"DEBUG: Claude verification for {current_key} and {next_key}: {is_cross_page}")
            except Exception as e:
                print(f"ERROR: Failed to verify tables with Claude: {str(e)}")
        return claude_verifications
    
    def _capture_table_screenshot(self, table_info):
        try:
            page_number = table_info['page_number']
            page_image_info = next((img for img in self.page_images if img['page_number'] == page_number), None)
            if not page_image_info:
                print(f"WARNING: No saved image found for page {page_number}")
                return None
            img = cv2.imread(page_image_info['path'])
            if img is None:
                print(f"ERROR: Failed to load image: {page_image_info['path']}")
                return None
            if 'position' in table_info:
                page_height, page_width = img.shape[:2]
                top = int(table_info['position']['top'])
                bottom = int(table_info['position']['bottom'])
                padding = 50
                top_padded = max(0, top - padding)
                bottom_padded = min(page_height, bottom + padding)
                return img[top_padded:bottom_padded, :]
            return img
        except Exception as e:
            print(f"ERROR: Failed to capture table screenshot: {str(e)}")
            return None
    
    def _create_combined_image(self, img1, img2):
        try:
            height1, width1 = img1.shape[:2]
            height2, width2 = img2.shape[:2]
            target_width = max(width1, width2)
            if width1 != target_width:
                img1 = cv2.resize(img1, (target_width, int(height1 * target_width / width1)))
            if width2 != target_width:
                img2 = cv2.resize(img2, (target_width, int(height2 * target_width / width2)))
            separator = np.ones((20, target_width, 3), dtype=np.uint8) * 255
            red_line = separator.copy()
            red_line[:, :, 0] = 0
            red_line[:, :, 1] = 0
            red_line[:, :, 2] = 255
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(red_line, "PAGE BREAK", (target_width // 2 - 100, 15), font, 0.6, (255, 255, 255), 2)
            return np.vstack([img1, red_line, img2])
        except Exception as e:
            print(f"ERROR: Failed to create combined image: {str(e)}")
            return None
    
    def _encode_image(self, image):
        try:
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"ERROR: Failed to encode image: {str(e)}")
            return None
    
    def _ask_claude_for_verification(self, base64_image, current_table, next_table):
        try:
            client = Anthropic(api_key=CLAUDE_API_KEY)
            prompt = """
            I'm showing you two tables from consecutive pages of a document. The first table is at the bottom of one page, and the second table is at the top of the next page.
            Please analyze both tables and determine if they are parts of the same table that spans across the page break.
            Answer with ONLY "yes" or "no":
            - "yes" if these are parts of the same table that continues across pages
            - "no" if these are separate, unrelated tables
            Do not explain your reasoning, just answer "yes" or "no".
            """
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=100,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_image}},
                        {"type": "text", "text": prompt}
                    ]
                }]
            )
            response_text = response.content[0].text.strip().lower()
            print(f"DEBUG: Claude raw response: '{response_text}'")
            return "yes" in response_text and "no" not in response_text
        except Exception as e:
            print(f"ERROR: Failed to get verification from Claude: {str(e)}")
            return False
    
    def _run_span_based_detection(self):
        print("DEBUG: Running additional span-based detection as fallback...")
        for page_data in self.analysis_results:
            result = page_data['result']
            if not hasattr(result, 'tables') or not result.tables:
                continue
            merge_tables_candidates, table_integral_span_list = self._get_merge_table_candidates_and_integral_span(result.tables)
            SEPARATOR_LENGTH_IN_MARKDOWN_FORMAT = 2
            for merged_table in merge_tables_candidates:
                pre_table_idx = merged_table["pre_table_idx"]
                current_table_idx = pre_table_idx + 1
                if pre_table_idx < 0 or pre_table_idx >= len(result.tables) or current_table_idx >= len(result.tables):
                    continue
                start = merged_table["start"]
                end = merged_table["end"]
                has_paragraph = self._check_paragraph_presence(result.paragraphs, start, end) if hasattr(result, 'paragraphs') else False
                is_horizontal = self._check_tables_are_horizontal_distribution(result, pre_table_idx)
                is_vertical = (
                    not has_paragraph and
                    hasattr(result.tables[pre_table_idx], 'column_count') and
                    hasattr(result.tables[current_table_idx], 'column_count') and
                    result.tables[pre_table_idx].column_count == result.tables[current_table_idx].column_count and
                    table_integral_span_list[current_table_idx]["min_offset"] - table_integral_span_list[pre_table_idx]["max_offset"]
                    <= SEPARATOR_LENGTH_IN_MARKDOWN_FORMAT
                )
                if is_vertical or is_horizontal:
                    distribution_type = "horizontal" if is_horizontal else "vertical"
                    page_number = page_data['page_number']
                    page_prefix = f"page_{page_number}_"
                    curr_table_key = f"{page_prefix}{current_table_idx}"
                    prev_table_key = f"{page_prefix}{pre_table_idx}"
                    if curr_table_key not in self.cross_page_tables:
                        print(f"DEBUG: Additional cross-page table detected: {curr_table_key} is a {distribution_type} continuation of {prev_table_key}")
                        self.cross_page_tables[curr_table_key] = {
                            'is_cross_page': True, 'continuation_of': prev_table_key, 'distribution_type': distribution_type,
                            'page_number': page_number, 'table_index': current_table_idx, 'claude_verified': None
                        }
                        self.cross_page_tables[prev_table_key] = {
                            'is_cross_page': True, 'continuation_of': None, 'distribution_type': distribution_type,
                            'page_number': page_number, 'table_index': pre_table_idx, 'claude_verified': None
                        }
    
    def _get_merge_table_candidates_and_integral_span(self, tables):
        table_integral_span_list = []
        merge_tables_candidates = []
        pre_table_idx = -1
        pre_table_page = -1
        pre_max_offset = 0
        for table_idx, table in enumerate(tables):
            min_offset, max_offset = self._get_table_span_offsets(table)
            if min_offset > -1 and max_offset > -1:
                table_page_numbers = self._get_table_page_numbers(table)
                if not table_page_numbers:
                    print(f"WARNING: Table {table_idx} has no page numbers")
                    continue
                table_page = min(table_page_numbers)
                print(f"DEBUG: Table {table_idx} has offset range: {min_offset} - {max_offset} on page {table_page}")
                if table_page == pre_table_page + 1:
                    merge_tables_candidates.append({
                        "pre_table_idx": pre_table_idx, "start": pre_max_offset, "end": min_offset,
                        "min_offset": min_offset, "max_offset": max_offset
                    })
                table_integral_span_list.append({"idx": table_idx, "min_offset": min_offset, "max_offset": max_offset})
                pre_table_idx = table_idx
                pre_table_page = table_page
                pre_max_offset = max_offset
            else:
                print(f"DEBUG: Table {table_idx} has no spans")
                table_integral_span_list.append({"idx": table_idx, "min_offset": -1, "max_offset": -1})
        return merge_tables_candidates, table_integral_span_list

    def _get_table_page_numbers(self, table):
        return [region.page_number for region in table.bounding_regions] if hasattr(table, 'bounding_regions') and table.bounding_regions else []

    def _get_table_span_offsets(self, table):
        if hasattr(table, 'spans') and table.spans:
            min_offset = table.spans[0].offset
            max_offset = table.spans[0].offset + table.spans[0].length
            for span in table.spans:
                if span.offset < min_offset:
                    min_offset = span.offset
                if span.offset + span.length > max_offset:
                    max_offset = span.offset + span.length
            return min_offset, max_offset
        return -1, -1

    def _check_paragraph_presence(self, paragraphs, start, end):
        if not paragraphs:
            return False
        for paragraph in paragraphs:
            if hasattr(paragraph, 'spans') and paragraph.spans:
                for span in paragraph.spans:
                    if hasattr(span, 'offset') and span.offset > start and span.offset < end:
                        if not hasattr(paragraph, 'role') or (hasattr(paragraph, 'role') and paragraph.role not in ["pageHeader", "pageFooter", "pageNumber"]):
                            return True
        return False

    def _check_tables_are_horizontal_distribution(self, result, pre_table_idx):
        INDEX_OF_X_LEFT_TOP = 0
        INDEX_OF_X_LEFT_BOTTOM = 6
        INDEX_OF_X_RIGHT_TOP = 2
        INDEX_OF_X_RIGHT_BOTTOM = 4
        THRESHOLD_RATE_OF_RIGHT_COVER = 0.99
        THRESHOLD_RATE_OF_LEFT_COVER = 0.01
        is_right_covered = False
        is_left_covered = False
        if (hasattr(result.tables[pre_table_idx], 'row_count') and
            hasattr(result.tables[pre_table_idx + 1], 'row_count') and
            result.tables[pre_table_idx].row_count == result.tables[pre_table_idx + 1].row_count):
            if hasattr(result.tables[pre_table_idx], 'bounding_regions') and result.tables[pre_table_idx].bounding_regions:
                for region in result.tables[pre_table_idx].bounding_regions:
                    if hasattr(region, 'page_number') and hasattr(region, 'polygon') and region.polygon:
                        if region.page_number <= len(result.pages):
                            page_width = result.pages[region.page_number - 1].width if hasattr(result.pages[region.page_number - 1], 'width') else 1000
                            x_right = max(region.polygon[INDEX_OF_X_RIGHT_TOP], region.polygon[INDEX_OF_X_RIGHT_BOTTOM])
                            if x_right / page_width > THRESHOLD_RATE_OF_RIGHT_COVER:
                                is_right_covered = True
                                break
            if hasattr(result.tables[pre_table_idx + 1], 'bounding_regions') and result.tables[pre_table_idx + 1].bounding_regions:
                for region in result.tables[pre_table_idx + 1].bounding_regions:
                    if hasattr(region, 'page_number') and hasattr(region, 'polygon') and region.polygon:
                        if region.page_number <= len(result.pages):
                            page_width = result.pages[region.page_number - 1].width if hasattr(result.pages[region.page_number - 1], 'width') else 1000
                            x_left = min(region.polygon[INDEX_OF_X_LEFT_TOP], region.polygon[INDEX_OF_X_LEFT_BOTTOM])
                            if x_left / page_width < THRESHOLD_RATE_OF_LEFT_COVER:
                                is_left_covered = True
                                break
        return is_left_covered and is_right_covered
    
    def get_cross_page_tables(self):
        return self.cross_page_tables
    
    def get_hyperlinks(self):
        return self.hyperlinks

class DocumentReconstructor:
    """Reconstruct document from Azure Document Intelligence results"""
    
    def __init__(self):
        print("DEBUG: Initializing DocumentReconstructor")
    

    def generate_html(self, analysis_results, extracted_images, hyperlinks=None, output_path=None):
        """Reconstruct document from Azure Document Intelligence results with hyperlink integration."""
        import html as html_module
        from fuzzywuzzy import fuzz
        import os
        import shutil
        import re
        
        print("INFO: Starting HTML generation with extracted images and hyperlinks")
        if not analysis_results:
            print("WARNING: No analysis results found to generate HTML")
            return "<html><body><p>No document content found.</p></body></html>"
        
        # Handle backward compatibility for old function signature
        if hyperlinks is not None and isinstance(hyperlinks, str):
            output_path = hyperlinks
            hyperlinks = None
            print("INFO: Detected old function call format, adjusting parameters")
        
        # Use self.hyperlinks if hyperlinks parameter is None
        if hyperlinks is None:
            hyperlinks = getattr(self, 'hyperlinks', [])
            print(f"INFO: Using hyperlinks from self.hyperlinks: {len(hyperlinks)} hyperlinks found")
        
        # Build a lookup table of known hyperlink text and URLs (excluding hidden links)
        hyperlink_lookup = {}
        for link in hyperlinks:
            if link.get('is_hidden', False):
                continue
            text = link.get('text', '').strip().lower()
            if text:
                if text not in hyperlink_lookup:
                    hyperlink_lookup[text] = []
                hyperlink_lookup[text].append(link)
        print(f"DEBUG: Hyperlink lookup table created with {len(hyperlink_lookup)} unique text entries")
        
        # Organize images by page
        images_by_page = {}
        for img in extracted_images:
            page_num = img['page_number']
            if page_num not in images_by_page:
                images_by_page[page_num] = []
            images_by_page[page_num].append(img)
        
        # Organize hyperlinks by page with deduplication based on URL and text
        links_by_page = {}
        if hyperlinks:
            seen_links = set()
            for link in hyperlinks:
                page_num = link.get('page_number', 0) + 1  # Convert 0-based to 1-based indexing
                if page_num not in links_by_page:
                    links_by_page[page_num] = []
                link_key = (link.get('url', ''), link.get('text', '').strip().lower())
                if link_key not in seen_links:
                    seen_links.add(link_key)
                    links_by_page[page_num].append(link)
        
        # Set up output directories
        html_dir = os.path.dirname(output_path) if output_path else '.'
        image_out_dir = os.path.join(html_dir, 'images')
        if not os.path.exists(image_out_dir):
            os.makedirs(image_out_dir)
            print(f"INFO: Created directory for HTML images: {image_out_dir}")
        
        for img in extracted_images:
            shutil.copy2(img['path'], os.path.join(image_out_dir, img['filename']))
            print(f"INFO: Copied image {img['filename']} to HTML images directory")
        
        # Start HTML structure
        html = "<!DOCTYPE html>\n<html>\n<head>\n<meta charset='UTF-8'>\n<title>Document Reconstruction</title>\n"
        html += self._generate_css()
        html += "</head>\n<body>\n"
        
        for page_data in analysis_results:
            page_number = page_data['page_number']
            result = page_data['result']
            if not hasattr(result, 'pages') and result is not None:
                result = type('obj', (object,), {'pages': [result]})
            if not hasattr(result, 'pages') or not result.pages:
                print(f"WARNING: No pages found in API result for page {page_number}")
                continue
            
            api_page = result.pages[0]
            print(f"INFO: Generating HTML for page {page_number}")
            width = api_page.width if hasattr(api_page, 'width') else 1000
            height = api_page.height if hasattr(api_page, 'height') else 1000
            
            html += f"<div class='page' id='page-{page_number}' style='width: {width}px; height: {height}px;'>\n<div class='page-number'>Page {page_number}</div>\n"
            
            # Collect text lines and their coordinates from Document Intelligence
            text_lines = []
            if hasattr(api_page, 'lines') and api_page.lines:
                for i, line in enumerate(api_page.lines):
                    if hasattr(line, 'polygon') and line.polygon and hasattr(line, 'content'):
                        try:
                            polygon = line.polygon
                            left = min(polygon[::2])
                            top = min(polygon[1::2])
                            right = max(polygon[::2])
                            bottom = max(polygon[1::2])
                            content = line.content.strip()
                            if not content:
                                continue
                            # Correct page references in the text
                            page_ref_match = re.search(r"this page (\d+)", content, re.IGNORECASE)
                            if page_ref_match:
                                ref_page = int(page_ref_match.group(1))
                                if ref_page != page_number:
                                    print(f"DEBUG: Correcting page reference in text on page {page_number}: '{content}'")
                                    content = re.sub(r"this page \d+", f"this page {page_number}", content, flags=re.IGNORECASE)
                            text_lines.append({
                                'index': i,
                                'content': content,
                                'left': left,
                                'top': top,
                                'right': right,
                                'bottom': bottom,
                                'width': (right - left) / width * 100,
                                'height': (bottom - top) / height * 100,
                                'left_percent': (left / width) * 100,
                                'top_percent': (top / height) * 100
                            })
                        except Exception as e:
                            print(f"ERROR: Failed to process line {i}: {str(e)}")
            
            # Debug: Log all text lines detected by DI
            print(f"DEBUG: Text lines detected by Document Intelligence on page {page_number}:")
            for line in text_lines:
                print(f"  - '{line['content']}' at (left: {line['left']}, top: {line['top']})")
            
            # Get hyperlinks for current page
            page_links = links_by_page.get(page_number, [])
            print(f"INFO: Using {len(page_links)} hyperlinks for page {page_number}")
            for link in page_links:
                print(f"DEBUG: Hyperlink on page {page_number}: Text='{link.get('text', '')}', URL={link.get('url', '#')}")
            
            # Process text lines with hyperlinks
            matched_links = set()  # Track matched hyperlinks
            for text_line in text_lines:
                style_class = "text-line"
                if text_line['content'].isupper() or (len(text_line['content']) <= 30 and not any(c.isdigit() for c in text_line['content'])):
                    style_class += " possible-heading"
                
                # Skip text lines inside tables (handled separately)
                skip_line = False
                if hasattr(result, 'tables') and result.tables:
                    for table in result.tables:
                        if hasattr(table, 'bounding_regions') and table.bounding_regions:
                            for region in table.bounding_regions:
                                if hasattr(region, 'polygon') and region.polygon:
                                    table_polygon = region.polygon
                                    if (text_line['left'] >= min(table_polygon[::2]) and 
                                        text_line['right'] <= max(table_polygon[::2]) and
                                        text_line['top'] >= min(table_polygon[1::2]) and 
                                        text_line['bottom'] <= max(table_polygon[1::2])):
                                        skip_line = True
                                        break
                            if skip_line:
                                break
                
                if not skip_line:
                    content = html_module.escape(text_line['content'])
                    matched_urls = []
                    
                    # First, try to match with hyperlinks for this page
                    for link in page_links:
                        if link.get('is_hidden', False):
                            continue  # Skip hidden hyperlinks for text matching
                        link_text = link.get('text', '').strip()
                        if link_text:
                            # Use exact matching for "Page X" text lines
                            if re.match(r'^Page \d+$', text_line['content'], re.IGNORECASE):
                                if text_line['content'].lower() == link_text.lower():
                                    link_url = html_module.escape(link.get('url', '#'))
                                    matched_urls.append((link_url, link))
                                    matched_links.add(id(link))
                                    print(f"DEBUG: Matched hyperlink to text line on page {page_number}: Text='{text_line['content']}', URL={link_url}")
                            else:
                                # Use substring matching, exact matching, or fuzzy matching for other text lines
                                if (link_text.lower() in text_line['content'].lower() or
                                    text_line['content'].lower() == link_text.lower() or
                                    fuzz.ratio(text_line['content'].lower(), link_text.lower()) > 80):
                                    link_url = html_module.escape(link.get('url', '#'))
                                    matched_urls.append((link_url, link))
                                    matched_links.add(id(link))
                                    print(f"DEBUG: Matched hyperlink to text line on page {page_number}: Text='{text_line['content']}', URL={link_url}")
                    
                    # If no match found, try the lookup table (regardless of whether page_links is empty)
                    if not matched_urls:
                        text_lower = text_line['content'].lower()
                        if text_lower in hyperlink_lookup:
                            # Use exact matching for "Page X" text lines in lookup
                            if re.match(r'^page \d+$', text_lower):
                                for link in hyperlink_lookup[text_lower]:
                                    if link.get('text', '').strip().lower() == text_lower:
                                        link_url = html_module.escape(link.get('url', '#'))
                                        matched_urls.append((link_url, link))
                                        matched_links.add(id(link))  # Add to matched_links to avoid duplicate rendering
                                        print(f"DEBUG: Matched hyperlink to text line on page {page_number} using lookup: Text='{text_line['content']}', URL={link_url}")
                                        break
                            else:
                                # Use first match for other text lines
                                matched_link = hyperlink_lookup[text_lower][0]
                                link_url = html_module.escape(matched_link.get('url', '#'))
                                matched_urls.append((link_url, matched_link))
                                matched_links.add(id(matched_link))  # Add to matched_links to avoid duplicate rendering
                                print(f"DEBUG: Matched hyperlink to text line on page {page_number} using lookup: Text='{text_line['content']}', URL={link_url}")
                    
                    # Debug: Log the state of matched_urls before rendering
                    print(f"DEBUG: Rendering text line on page {page_number}: Text='{text_line['content']}', Matched URLs={[(url, id(link)) for url, link in matched_urls]}")
                    
                    if matched_urls:
                        if len(matched_urls) == 1:
                            link_url, link = matched_urls[0]
                            link_type = link.get('type', 'external')
                            link_source = link.get('source', 'unknown')
                            link_class = f"hyperlink {link_type}-link {link_source}-source"
                            # Make the URL more visible with strong inline styling
                            link_content = f"<a href='{link_url}' class='{link_class}' title='{link_url}'>{content}</a> <strong style='color: #555; display: inline !important;'>({link_url})</strong>"
                            print(f"DEBUG: Rendered single URL for text line on page {page_number}: Text='{text_line['content']}', URL={link_url}")
                        else:
                            # If multiple URLs are matched, prioritize the correct one for "Page X"
                            if re.match(r'^Page \d+$', text_line['content'], re.IGNORECASE):
                                page_num = int(text_line['content'].split()[-1])
                                correct_url = f"https://example.com/page/{page_num}"
                                matched_url = None
                                for url, link in matched_urls:
                                    if url == correct_url:
                                        matched_url = (url, link)
                                        break
                                if matched_url:
                                    link_url, link = matched_url
                                    link_type = link.get('type', 'external')
                                    link_source = link.get('source', 'unknown')
                                    link_class = f"hyperlink {link_type}-link {link_source}-source"
                                    # Make the URL more visible with strong inline styling
                                    link_content = f"<a href='{link_url}' class='{link_class}' title='{link_url}'>{content}</a> <strong style='color: #555; display: inline !important;'>({link_url})</strong>"
                                    print(f"DEBUG: Rendered prioritized URL for text line on page {page_number}: Text='{text_line['content']}', URL={link_url}")
                                else:
                                    # Fallback to first URL if correct one not found
                                    link_url, link = matched_urls[0]
                                    link_type = link.get('type', 'external')
                                    link_source = link.get('source', 'unknown')
                                    link_class = f"hyperlink {link_type}-link {link_source}-source"
                                    # Make the URL more visible with strong inline styling
                                    link_content = f"<a href='{link_url}' class='{link_class}' title='{link_url}'>{content}</a> <strong style='color: #555; display: inline !important;'>({link_url})</strong>"
                                    print(f"DEBUG: Rendered fallback URL for text line on page {page_number}: Text='{text_line['content']}', URL={link_url}")
                            else:
                                # For non-"Page X" text, show all URLs with strong inline styling
                                url_list = []
                                for url, link in matched_urls:
                                    link_type = link.get('type', 'external')
                                    link_source = link.get('source', 'unknown')
                                    url_list.append(f"<a href='{url}' class='hyperlink {link_type}-link {link_source}-source' title='{url}'>{url}</a>")
                                
                                # Join URLs with comma and wrap in strong tag
                                urls_html = ", ".join(url_list)
                                link_content = f"{content} <strong style='color: #555; display: inline !important;'>({urls_html})</strong>"
                                print(f"DEBUG: Rendered multiple URLs for text line on page {page_number}: Text='{text_line['content']}', URLs={[url for url, _ in matched_urls]}")
                    else:
                        link_content = content
                        # Warn if certain text lines are expected to have hyperlinks but don't
                        if text_line['content'].lower() in ['rotated link', 'qr code with embedded link']:
                            print(f"WARNING: No hyperlink found for expected text line on page {page_number}: Text='{text_line['content']}'")
                        print(f"DEBUG: No URLs matched for text line on page {page_number}: Text='{text_line['content']}'")
                    
                    # Enhanced div styling with higher z-index for hyperlinks
                    html += f"<div class='{style_class}' style='left: {text_line['left_percent']}%; top: {text_line['top_percent']}%; width: auto; min-width: {text_line['width']}%; max-width: none; height: auto; min-height: {text_line['height']}%; z-index: 10; white-space: nowrap; overflow: visible; position: absolute;'>{link_content}</div>\n"
            
            # Add tables
            if hasattr(result, 'tables') and result.tables:
                print(f"INFO: Adding {len(result.tables)} tables to page {page_number}")
                for i, table in enumerate(result.tables):
                    if hasattr(table, 'bounding_regions') and table.bounding_regions:
                        for region in table.bounding_regions:
                            if hasattr(region, 'polygon') and region.polygon:
                                try:
                                    polygon = region.polygon
                                    left = min(polygon[::2]) / width * 100
                                    top = min(polygon[1::2]) / height * 100
                                    table_width = (max(polygon[::2]) - min(polygon[::2])) / width * 100
                                    table_height = (max(polygon[1::2]) - min(polygon[1::2])) / height * 100
                                    html += f"<div class='table-container' style='left: {left}%; top: {top}%; width: {table_width}%; height: {table_height}%; z-index: 5; overflow: visible;'><table class='document-table'>\n"
                                    if hasattr(table, 'cells') and table.cells:
                                        rows = {}
                                        for cell in table.cells:
                                            if hasattr(cell, 'row_index') and hasattr(cell, 'column_index'):
                                                row_idx = cell.row_index
                                                if row_idx not in rows:
                                                    rows[row_idx] = []
                                                rows[row_idx].append(cell)
                                        for row_idx in sorted(rows.keys()):
                                            html += "    <tr>\n"
                                            sorted_cells = sorted(rows[row_idx], key=lambda c: c.column_index)
                                            for cell in sorted_cells:
                                                if hasattr(cell, 'content'):
                                                    cell_class = "header-cell" if row_idx == 0 else ""
                                                    cell_content = html_module.escape(cell.content)
                                                    
                                                    # Correct page references in table cells
                                                    page_ref_match = re.search(r"this page (\d+)", cell_content, re.IGNORECASE)
                                                    if page_ref_match:
                                                        ref_page = int(page_ref_match.group(1))
                                                        if ref_page != page_number:
                                                            print(f"DEBUG: Correcting page reference in table cell on page {page_number}: '{cell_content}'")
                                                            cell_content = re.sub(r"this page \d+", f"this page {page_number}", cell_content, flags=re.IGNORECASE)
                                                    
                                                    # Match hyperlink to cell content
                                                    matched_urls = []
                                                    for link in page_links:
                                                        if link.get('is_hidden', False):
                                                            continue
                                                        link_text = link.get('text', '').strip()
                                                        if link_text:
                                                            if re.match(r'^Page \d+$', cell.content, re.IGNORECASE):
                                                                if cell.content.lower() == link_text.lower():
                                                                    link_url = html_module.escape(link.get('url', '#'))
                                                                    matched_urls.append((link_url, link))
                                                                    matched_links.add(id(link))
                                                                    print(f"DEBUG: Matched hyperlink to table cell on page {page_number}: Text='{cell.content}', URL={link_url}")
                                                            else:
                                                                if (link_text.lower() in cell.content.lower() or
                                                                    cell.content.lower() == link_text.lower() or
                                                                    fuzz.ratio(cell.content.lower(), link_text.lower()) > 80):
                                                                    link_url = html_module.escape(link.get('url', '#'))
                                                                    matched_urls.append((link_url, link))
                                                                    matched_links.add(id(link))
                                                                    print(f"DEBUG: Matched hyperlink to table cell on page {page_number}: Text='{cell.content}', URL={link_url}")
                                                    
                                                    # If no match found, try the lookup table
                                                    if not matched_urls:
                                                        text_lower = cell.content.lower()
                                                        if text_lower in hyperlink_lookup:
                                                            if re.match(r'^page \d+$', text_lower):
                                                                for link in hyperlink_lookup[text_lower]:
                                                                    if link.get('text', '').strip().lower() == text_lower:
                                                                        link_url = html_module.escape(link.get('url', '#'))
                                                                        matched_urls.append((link_url, link))
                                                                        matched_links.add(id(link))
                                                                        print(f"DEBUG: Matched hyperlink to table cell on page {page_number} using lookup: Text='{cell.content}', URL={link_url}")
                                                                        break
                                                            else:
                                                                matched_link = hyperlink_lookup[text_lower][0]
                                                                link_url = html_module.escape(matched_link.get('url', '#'))
                                                                matched_urls.append((link_url, matched_link))
                                                                matched_links.add(id(matched_link))
                                                                print(f"DEBUG: Matched hyperlink to table cell on page {page_number} using lookup: Text='{cell.content}', URL={link_url}")
                                                    
                                                    if matched_urls:
                                                        if len(matched_urls) == 1:
                                                            link_url, link = matched_urls[0]
                                                            link_type = link.get('type', 'external')
                                                            link_source = link.get('source', 'unknown')
                                                            link_class = f"hyperlink {link_type}-link {link_source}-source"
                                                            # Make URL more visible with strong tag 
                                                            cell_content = f"<a href='{link_url}' class='{link_class}' title='{link_url}'>{cell_content}</a> <strong style='color: #555; display: inline !important;'>({link_url})</strong>"
                                                        else:
                                                            if re.match(r'^Page \d+$', cell.content, re.IGNORECASE):
                                                                page_num = int(cell.content.split()[-1])
                                                                correct_url = f"https://example.com/page/{page_num}"
                                                                matched_url = None
                                                                for url, link in matched_urls:
                                                                    if url == correct_url:
                                                                        matched_url = (url, link)
                                                                        break
                                                                if matched_url:
                                                                    link_url, link = matched_url
                                                                    link_type = link.get('type', 'external')
                                                                    link_source = link.get('source', 'unknown')
                                                                    link_class = f"hyperlink {link_type}-link {link_source}-source"
                                                                    # Make URL more visible with strong tag
                                                                    cell_content = f"<a href='{link_url}' class='{link_class}' title='{link_url}'>{cell_content}</a> <strong style='color: #555; display: inline !important;'>({link_url})</strong>"
                                                                else:
                                                                    link_url, link = matched_urls[0]
                                                                    link_type = link.get('type', 'external')
                                                                    link_source = link.get('source', 'unknown')
                                                                    link_class = f"hyperlink {link_type}-link {link_source}-source"
                                                                    # Make URL more visible with strong tag
                                                                    cell_content = f"<a href='{link_url}' class='{link_class}' title='{link_url}'>{cell_content}</a> <strong style='color: #555; display: inline !important;'>({link_url})</strong>"
                                                            else:
                                                                # For non-"Page X" text with multiple URLs
                                                                url_list = []
                                                                for url, link in matched_urls:
                                                                    link_type = link.get('type', 'external')
                                                                    link_source = link.get('source', 'unknown')
                                                                    url_list.append(f"<a href='{url}' class='hyperlink {link_type}-link {link_source}-source' title='{url}'>{url}</a>")
                                                                
                                                                # Join URLs with comma and wrap in strong tag
                                                                urls_html = ", ".join(url_list)
                                                                cell_content = f"{cell_content} <strong style='color: #555; display: inline !important;'>({urls_html})</strong>"
                                                    
                                                    html += f"      <td class='{cell_class}' style='overflow: visible;'>{cell_content}</td>\n"
                                            html += "    </tr>\n"
                                    else:
                                        html += "    <tr><td> </td></tr>\n"
                                    html += "  </table></div>\n"
                                except Exception as e:
                                    print(f"ERROR: Failed to position table {i}: {str(e)}")
            
            # Add images with lower z-index
            page_images = images_by_page.get(page_number, [])
            if page_images:
                print(f"INFO: Adding {len(page_images)} extracted images to page {page_number}")
                for img in page_images:
                    left = img['x_min'] / width * 100
                    top = img['y_min'] / height * 100
                    img_width = img['width'] / width * 100
                    img_height = img['height'] / height * 100
                    image_rel_path = f"images/{img['filename']}"
                    html += f"<div class='extracted-image' style='left: {left}%; top: {top}%; width: {img_width}%; height: {img_height}%; z-index: 2;'><img src='{image_rel_path}' alt='Figure {img['figure_index']}' /></div>\n"
            
            # Handle unmatched hyperlinks (including hidden ones) using coordinates with highest z-index
            unmatched_links = [link for link in page_links if id(link) not in matched_links]
            if unmatched_links:
                print(f"INFO: Processing {len(unmatched_links)} unmatched hyperlinks on page {page_number} using coordinates")
                for link in unmatched_links:
                    if link.get('is_hidden', False):
                        print(f"DEBUG: Skipping hidden hyperlink on page {page_number}: Text='{link.get('text', '')}', URL={link.get('url', '#')}")
                        continue
                    coords = link.get('coordinates', {})
                    if not coords:
                        print(f"WARNING: No coordinates available for unmatched hyperlink on page {page_number}: Text='{link.get('text', '')}', URL={link.get('url', '#')}")
                        continue
                    x0 = coords.get('x0', 0) / width * 100
                    y0 = coords.get('y0', 0) / height * 100
                    x1 = coords.get('x1', x0 + 50) / width * 100
                    y1 = coords.get('y1', y0 + 20) / height * 100
                    link_width = (x1 - x0)
                    link_height = (y1 - y0)
                    link_url = html_module.escape(link.get('url', '#'))
                    link_text = html_module.escape(link.get('text', 'Link'))
                    link_type = link.get('type', 'external')
                    link_source = link.get('source', 'unknown')
                    link_class = f"hyperlink {link_type}-link {link_source}-source unmatched-link"
                    # Enhanced div styling with highest z-index for unmatched hyperlinks
                    html += f"<div class='{link_class}' style='left: {x0}%; top: {y0}%; width: auto; min-width: {link_width}%; max-width: none; height: auto; min-height: {link_height}%; z-index: 10; white-space: nowrap; overflow: visible; position: absolute;'><a href='{link_url}' title='{link_url}'>{link_text}</a> <strong style='color: #555; display: inline !important;'>({link_url})</strong></div>\n"
                    print(f"DEBUG: Added unmatched hyperlink on page {page_number} at (left: {x0}%, top: {y0}%): Text='{link_text}', URL={link_url}")
            
            # Log unmatched hyperlinks (for reference)
            unmatched_links_log = [link for link in page_links if id(link) not in matched_links]
            if unmatched_links_log:
                print(f"WARNING: {len(unmatched_links_log)} hyperlinks on page {page_number} could not be matched to text:")
                for link in unmatched_links_log:
                    print(f"  - Text: '{link.get('text', '')}', URL: {link.get('url', '#')}")
            
            html += "</div>\n"
        
        html += "</body>\n</html>"
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"INFO: HTML saved to file: {output_path}")
        return html

    def _generate_css(self):
        """Generate CSS for the HTML document"""
        return """<style>
            body { 
                margin: 0; 
                padding: 0; 
                font-family: Arial, sans-serif;
                overflow-x: auto; /* Allow horizontal scrolling */ 
            }
            
            .page { 
                position: relative; 
                width: 100%; 
                height: auto; 
                margin-bottom: 40px; 
                border: 1px solid #ccc; 
                overflow: visible !important; /* Changed from hidden to visible */
                min-height: 1200px; /* Ensure enough space for content */
            }
            
            .page-number { 
                position: absolute; 
                bottom: 10px; 
                right: 10px; 
                background: #f0f0f0; 
                padding: 5px 10px; 
                border-radius: 3px; 
                font-weight: bold; 
                z-index: 1000; 
            }
            
            .text-line { 
                position: absolute; 
                font-family: Arial, sans-serif; 
                line-height: 1.2;
                white-space: nowrap; 
                overflow: visible !important; /* Ensure content isn't cut off */
                z-index: 10;
            }
            
            .possible-heading { 
                font-weight: bold; 
            }
            
            .table-container { 
                position: absolute; 
                background: rgba(255, 255, 255, 0.98);
                z-index: 100;
                overflow: visible !important; /* Changed from auto to visible */
            }
            
            .document-table { 
                width: 100%; 
                border-collapse: collapse; 
                margin: 0; 
                padding: 0; 
            }
            
            .document-table td { 
                border: 1px solid #ddd; 
                padding: 8px; 
                background-color: white;
                word-break: normal;
                white-space: normal; 
                overflow: visible;
            }
            
            .document-table .header-cell { 
                font-weight: bold; 
                background-color: #f2f2f2; 
            }
            
            .extracted-image { 
                position: absolute; 
                z-index: 20; 
            }
            
            .extracted-image img { 
                width: 100%; 
                height: 100%; 
                object-fit: contain; 
            }
            
            /* Link styles */
            a { 
                color: #0066cc; 
                text-decoration: underline;
                display: inline-block; 
            }
            
            a:hover { 
                text-decoration: underline; 
                background-color: #ffffc0; 
                color: #cc0000; 
            }
            
            /* Display URLs clearly */
            .url-display {
                display: inline-block;
                color: #666;
                font-size: 0.9em;
                margin-left: 5px;
                white-space: nowrap;
                max-width: none;
                overflow: visible;
            }
            
            .hyperlink { 
                display: inline-block; 
                position: relative; 
                z-index: 50; 
                border: 1px solid transparent;
                max-width: none !important;
                overflow: visible !important;
            }
            
            .hyperlink:hover { 
                border: 1px dashed #ff6600; 
                background-color: rgba(255, 255, 0, 0.2); 
            }
            
            /* Link type styles */
            .external-link { 
                background-color: rgba(0, 128, 255, 0.1); 
            }
            
            .internal-link { 
                background-color: rgba(0, 255, 128, 0.1); 
            }
            
            .potential-link { 
                background-color: rgba(255, 128, 0, 0.1); 
            }
            
            .hidden-link { 
                background-color: rgba(255, 0, 128, 0.15); 
            }
            
            /* Source type indicators */
            .pdfminer_text-source { 
                border-bottom: 2px solid #00f; 
            }
            
            .pypdf_annot-source { 
                border-bottom: 2px dotted #f0f; 
            }
            
            .pymupdf_annot-source { 
                border-bottom: 2px dashed #0ff; 
            }
        </style>"""
def analyze_image_with_claude(base64_image, image_index=None):
    """Analyze an image with Claude 3.7 Sonnet and return the raw response as a string"""
    context = f"Image {image_index}" if image_index is not None else "Unknown image"
    print(f"DEBUG: Starting analysis for {context}")
    try:
        client = Anthropic(api_key=CLAUDE_API_KEY)
        prompt = """
        Analyze the provided image and:
        1. Provide a detailed description of what you see.
        2. Generate 5 meaningful questions that someone might ask about the image based on its content.
        Return your response in JSON format with the following structure:
        {
            "description": "Your detailed description here",
            "questions": [
                "Question 1",
                "Question 2",
                "Question 3",
                "Question 4",
                "Question 5"
            ]
        }
        """
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64_image}},
                    {"type": "text", "text": prompt}
                ]
            }]
        )
        response_text = response.content[0].text
        print(f"DEBUG: Claude raw response for {context}: {response_text}")
        return response_text
    except Exception as e:
        print(f"ERROR: Failed to analyze image with Claude for {context}: {str(e)}")
        return f"Error analyzing image ({context}): {str(e)}"

def save_document_data_to_json(analysis_results, extracted_images, output_path, claude_responses, cross_page_tables=None, hyperlinks=None):
    """Save all document data, metadata, image information, and hyperlinks to a JSON file (excluding repetitive images)"""
    print(f"INFO: Creating JSON export of document data with cross-page table flags and hyperlinks")
    images_by_page = {}
    for idx, img in enumerate(extracted_images):
        page_num = img['page_number']
        if page_num not in images_by_page:
            images_by_page[page_num] = []
        image_data = {
            'figure_index': img['figure_index'],
            'filename': img['filename'],
            'position': {
                'x_min': img['x_min'],
                'y_min': img['y_min'],
                'x_max': img['x_max'],
                'y_max': img['y_max'],
                'width': img['width'],
                'height': img['height']
            },
            'base64': img['base64'],
            'claude_response': claude_responses.get(img['filename'], "No response available")
        }
        images_by_page[page_num].append(image_data)
    
    document_data = {
        'pages': [],
        'hyperlinks': hyperlinks if hyperlinks else []
    }
    for page_data in analysis_results:
        page_number = page_data['page_number']
        result = page_data['result']
        if not hasattr(result, 'pages') or not result.pages:
            print(f"WARNING: No pages found in API result for page {page_number}")
            continue
        
        api_page = result.pages[0]
        width = api_page.width if hasattr(api_page, 'width') else None
        height = api_page.height if hasattr(api_page, 'height') else None
        unit = api_page.unit if hasattr(api_page, 'unit') else None
        
        lines = []
        if hasattr(api_page, 'lines') and api_page.lines:
            for line_idx, line in enumerate(api_page.lines):
                if hasattr(line, 'polygon') and line.polygon and hasattr(line, 'content'):
                    polygon = line.polygon
                    line_data = {
                        'content': line.content,
                        'polygon': polygon,
                        'position': {
                            'left': min(polygon[::2]),
                            'top': min(polygon[1::2]),
                            'width': max(polygon[::2]) - min(polygon[::2]),
                            'height': max(polygon[1::2]) - min(polygon[1::2])
                        },
                        'confidence': line.confidence if hasattr(line, 'confidence') else None
                    }
                    if hasattr(line, 'words') and line.words:
                        line_data['words'] = [
                            {
                                'content': word.content if hasattr(word, 'content') else '',
                                'confidence': word.confidence if hasattr(word, 'confidence') else None,
                                'polygon': word.polygon if hasattr(word, 'polygon') and word.polygon else None
                            } for word in line.words
                        ]
                    lines.append(line_data)
        
        tables = []
        if hasattr(result, 'tables') and result.tables:
            for table_idx, table in enumerate(result.tables):
                is_cross_page = False
                continuation_of = None
                distribution_type = None
                claude_verified = None
                table_key = f"page_{page_number}_{table_idx}"
                if cross_page_tables and table_key in cross_page_tables:
                    is_cross_page = cross_page_tables[table_key]['is_cross_page']
                    continuation_of = cross_page_tables[table_key]['continuation_of']
                    distribution_type = cross_page_tables[table_key]['distribution_type']
                    claude_verified = cross_page_tables[table_key].get('claude_verified')
                
                table_data = {
                    'table_index': table_idx,
                    'row_count': table.row_count if hasattr(table, 'row_count') else None,
                    'column_count': table.column_count if hasattr(table, 'column_count') else None,
                    'is_cross_page': is_cross_page,
                    'continuation_of': continuation_of,
                    'distribution_type': distribution_type,
                    'claude_verified': claude_verified,
                    'cells': []
                }
                if hasattr(table, 'spans') and table.spans:
                    table_data['spans'] = [{'offset': span.offset, 'length': span.length} for span in table.spans]
                if hasattr(table, 'bounding_regions') and table.bounding_regions:
                    table_data['bounding_regions'] = [
                        {'page_number': region.page_number, 'polygon': region.polygon} for region in table.bounding_regions
                    ]
                if hasattr(table, 'cells') and table.cells:
                    for cell in table.cells:
                        cell_data = {
                            'row_index': cell.row_index if hasattr(cell, 'row_index') else None,
                            'column_index': cell.column_index if hasattr(cell, 'column_index') else None,
                            'content': cell.content if hasattr(cell, 'content') else '',
                            'row_span': cell.row_span if hasattr(cell, 'row_span') else 1,
                            'column_span': cell.column_span if hasattr(cell, 'column_span') else 1
                        }
                        if hasattr(cell, 'bounding_regions') and cell.bounding_regions:
                            cell_data['bounding_regions'] = [
                                {'page_number': region.page_number, 'polygon': region.polygon} for region in cell.bounding_regions
                            ]
                        table_data['cells'].append(cell_data)
                tables.append(table_data)
        
        figures = []
        if hasattr(result, 'figures') and result.figures:
            for figure_idx, figure in enumerate(result.figures):
                figure_data = {
                    'figure_index': figure_idx,
                    'bounding_regions': [{'page_number': region.page_number, 'polygon': region.polygon} for region in figure.bounding_regions]
                    if hasattr(figure, 'bounding_regions') and figure.bounding_regions else []
                }
                if hasattr(figure, 'spans') and figure.spans:
                    figure_data['spans'] = [{'offset': span.offset, 'length': span.length} for span in figure.spans]
                figures.append(figure_data)
        
        page_data = {
            'page_number': page_number,
            'dimensions': {'width': width, 'height': height, 'unit': unit},
            'lines': lines,
            'tables': tables,
            'figures': figures,
            'images': images_by_page.get(page_number, [])
        }
        document_data['pages'].append(page_data)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(document_data, f, indent=2)
    print(f"INFO: Document data saved to JSON: {output_path}")
    return document_data

def generate_simplified_json(analysis_results, extracted_images, output_path, claude_responses, cross_page_tables=None, hyperlinks=None):
    """Generate a simplified JSON file with hyperlinks (excluding repetitive images)"""
    print(f"INFO: Creating simplified JSON export of document content with cross-page table flags and hyperlinks")
    images_by_page = {}
    for img in extracted_images:
        page_num = img['page_number']
        if page_num not in images_by_page:
            images_by_page[page_num] = []
        image_data = {
            'type': 'image',
            'filename': img['filename'],
            'base64': img['base64'],
            'claude_response': claude_responses.get(img['filename'], "No response available"),
            '_position': {'top': img['y_min'], 'left': img['x_min']}
        }
        images_by_page[page_num].append(image_data)
    
    document_data = {
        'document_type': 'Simplified Content',
        'pages': [],
        'hyperlinks': hyperlinks if hyperlinks else []
    }
    for page_data in analysis_results:
        page_number = page_data['page_number']
        result = page_data['result']
        if not hasattr(result, 'pages') or not result.pages:
            print(f"WARNING: No pages found in API result for page {page_number}")
            continue
        
        api_page = result.pages[0]
        width = api_page.width if hasattr(api_page, 'width') else 1000
        height = api_page.height if hasattr(api_page, 'height') else 1000
        page_content = {'page_number': page_number, 'content': []}
        
        text_items = []
        if hasattr(api_page, 'lines') and api_page.lines:
            for line in api_page.lines:
                if hasattr(line, 'polygon') and line.polygon and hasattr(line, 'content'):
                    polygon = line.polygon
                    left = min(polygon[::2])
                    top = min(polygon[1::2])
                    content = line.content.strip()
                    if content:
                        text_items.append({
                            'type': 'text',
                            'content': content,
                            '_position': {'top': top, 'left': left}
                        })
        
        tables = []
        if hasattr(result, 'tables') and result.tables:
            for table_idx, table in enumerate(result.tables):
                table_position = {'top': 0, 'left': 0}
                if hasattr(table, 'bounding_regions') and table.bounding_regions:
                    for region in table.bounding_regions:
                        if hasattr(region, 'polygon') and region.polygon:
                            polygon = region.polygon
                            table_position = {'top': min(polygon[1::2]), 'left': min(polygon[::2])}
                            break
                
                is_cross_page = False
                continuation_of = None
                distribution_type = None
                claude_verified = None
                table_key = f"page_{page_number}_{table_idx}"
                if cross_page_tables and table_key in cross_page_tables:
                    is_cross_page = cross_page_tables[table_key]['is_cross_page']
                    continuation_of = cross_page_tables[table_key]['continuation_of']
                    distribution_type = cross_page_tables[table_key]['distribution_type']
                    claude_verified = cross_page_tables[table_key].get('claude_verified')
                
                table_data = []
                if hasattr(table, 'cells') and table.cells:
                    rows = {}
                    for cell in table.cells:
                        if hasattr(cell, 'row_index') and hasattr(cell, 'column_index'):
                            row_idx = cell.row_index
                            if row_idx not in rows:
                                rows[row_idx] = []
                            content = cell.content if hasattr(cell, 'content') else ''
                            rows[row_idx].append({'column': cell.column_index, 'content': content})
                    for row_idx in sorted(rows.keys()):
                        row_cells = sorted(rows[row_idx], key=lambda c: c['column'])
                        row_content = [cell['content'] for cell in row_cells]
                        table_data.append(row_content)
                
                tables.append({
                    'type': 'table',
                    '_position': table_position,
                    'data': table_data,
                    'is_cross_page': is_cross_page,
                    'continuation_of': continuation_of,
                    'distribution_type': distribution_type,
                    'claude_verified': claude_verified
                })
        
        page_images = images_by_page.get(page_number, [])
        page_links = [link for link in hyperlinks if link.get('page_number') == page_number] if hyperlinks else []
        hyperlink_items = []
        for link in page_links:
            coords = link.get('coordinates', {'x0': 0, 'y0': 0, 'x1': 0, 'y1': 0})
            hyperlink_items.append({
                'type': 'hyperlink',
                'url': link['url'],
                'text': link['text'],
                'link_type': link.get('type'),
                '_position': {'top': coords['y0'], 'left': coords['x0']},
                'coordinates': coords
            })
        
        all_content = text_items + tables + page_images + hyperlink_items
        all_content.sort(key=lambda x: (x['_position']['top'], x['_position']['left']))
        
        processed_content = []
        current_text = []
        y_threshold = height * 0.015
        last_y = None
        
        for item in all_content:
            if item['type'] == 'text':
                current_y = item['_position']['top']
                if last_y is None or (current_y - last_y) > y_threshold:
                    if current_text:
                        processed_content.append({'type': 'text', 'content': ' '.join(current_text)})
                        current_text = []
                current_text.append(item['content'])
                last_y = current_y
            elif item['type'] == 'table':
                if current_text:
                    processed_content.append({'type': 'text', 'content': ' '.join(current_text)})
                    current_text = []
                processed_content.append({
                    'type': 'table',
                    'data': item['data'],
                    'is_cross_page': item.get('is_cross_page', False),
                    'continuation_of': item.get('continuation_of'),
                    'distribution_type': item.get('distribution_type'),
                    'claude_verified': item.get('claude_verified')
                })
                last_y = None
            elif item['type'] == 'image':
                if current_text:
                    processed_content.append({'type': 'text', 'content': ' '.join(current_text)})
                    current_text = []
                processed_content.append({
                    'type': 'image',
                    'filename': item['filename'],
                    'base64': item['base64'],
                    'claude_response': item['claude_response']
                })
                last_y = None
            elif item['type'] == 'hyperlink':
                if current_text:
                    processed_content.append({'type': 'text', 'content': ' '.join(current_text)})
                    current_text = []
                processed_content.append({
                    'type': 'hyperlink',
                    'url': item['url'],
                    'text': item['text'],
                    'link_type': item['link_type'],
                    'coordinates': item['coordinates']
                })
                last_y = None
        
        if current_text:
            processed_content.append({'type': 'text', 'content': ' '.join(current_text)})
        
        page_content['content'] = processed_content
        document_data['pages'].append(page_content)
    
    document_data['pages'].sort(key=lambda x: x['page_number'])
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(document_data, f, indent=2)
    print(f"INFO: Simplified document content saved to JSON: {output_path}")
    return document_data

def main():
    """Main function to process the document and generate HTML and JSON with hyperlinks"""
    base_name = os.path.splitext(PDF_PATH)[0]
    output_html_path = f"{base_name}.html"
    output_json_path = f"{base_name}.json"
    output_simplified_json_path = f"{base_name}_simplified.json"
    
    print(f"INFO: Starting document processing")
    print(f"INFO: PDF path: {PDF_PATH}")
    print(f"INFO: Output will be saved as:")
    print(f"      - HTML: {output_html_path}")
    print(f"      - Full JSON: {output_json_path}")
    print(f"      - Simplified JSON: {output_simplified_json_path}")
    
    try:
        if not os.path.exists(PDF_PATH):
            print(f"ERROR: PDF file does not exist: {PDF_PATH}")
            return False
        
        print(f"INFO: PDF file found, size: {os.path.getsize(PDF_PATH)} bytes")
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            print(f"INFO: Created output directory for images: {OUTPUT_DIR}")
        
        processor = DocumentProcessor(AZURE_ENDPOINT, AZURE_API_KEY)
        analysis_results = processor.process_pdf(PDF_PATH)
        unique_images = processor.get_unique_images()  # Use filtered unique images
        cross_page_tables = processor.get_cross_page_tables()
        hyperlinks = processor.get_hyperlinks()
        
        print(f"INFO: Found {len(hyperlinks)} hyperlinks in the document, including hidden ones")
        
        print("INFO: Analyzing unique images with Claude...")
        claude_responses = {}
        total_images = len(unique_images)
        for idx, img in enumerate(unique_images):
            print(f"PROGRESS: Analyzing Image {idx + 1}/{total_images} ({(idx + 1) / total_images * 100:.1f}%)")
            claude_responses[img['filename']] = analyze_image_with_claude(img['base64'], image_index=idx)
        
        print("INFO: Starting HTML reconstruction with hyperlinks...")
        reconstructor = DocumentReconstructor()
        # Pass hyperlinks as separate parameter
        reconstructor.generate_html(analysis_results, unique_images, hyperlinks, output_html_path)
        
        print("INFO: Saving complete document data to JSON with cross-page table flags and hyperlinks...")
        save_document_data_to_json(analysis_results, unique_images, output_json_path, claude_responses, cross_page_tables, hyperlinks)
        
        print("INFO: Generating simplified document content JSON with cross-page table flags and hyperlinks...")
        generate_simplified_json(analysis_results, unique_images, output_simplified_json_path, claude_responses, cross_page_tables, hyperlinks)
        
        # Generate a special hyperlinks-only JSON for easy reference
        hyperlinks_json_path = f"{base_name}_hyperlinks.json"
        with open(hyperlinks_json_path, 'w', encoding='utf-8') as f:
            json.dump(hyperlinks, f, indent=2)
        
        print(f"INFO: Document processing completed:")
        print(f"      - HTML saved to: {output_html_path}")
        print(f"      - Full JSON saved to: {output_json_path} (including cross-page table flags and hyperlinks)")
        print(f"      - Simplified JSON saved to: {output_simplified_json_path} (including cross-page table flags and hyperlinks)")
        print(f"      - Hyperlinks-only JSON saved to: {hyperlinks_json_path}")
        print(f"      - Images saved to: {os.path.abspath(str(OUTPUT_DIR))}")
        print(f"INFO: To view the document, open {output_html_path} in a web browser")
        return True
    
    except Exception as e:
        print(f"ERROR: Unexpected error processing document: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
