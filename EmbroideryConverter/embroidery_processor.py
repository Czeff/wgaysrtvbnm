import os
import logging
import subprocess
import tempfile
from PIL import Image, ImageDraw, ImageFont
import pyembroidery
import cairosvg
import io
import base64
from xml.etree import ElementTree as ET
import math
import random

class EmbroideryProcessor:
    """Process images for embroidery pattern generation"""
    
    def __init__(self, output_folder):
        self.output_folder = output_folder
        self.stitch_colors = [
            '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
            '#800000', '#008000', '#000080', '#808000', '#800080', '#008080',
            '#C0C0C0', '#808080', '#FFA500', '#A52A2A', '#DEB887', '#5F9EA0'
        ]
    
    def process_file(self, file_path):
        """Process uploaded file and generate embroidery pattern"""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.svg':
                # Direct SVG processing
                svg_path = file_path
                processing_type = "SVG (direct processing)"
            else:
                # Convert raster to SVG
                svg_path = self._vectorize_image(file_path)
                processing_type = f"Raster to SVG conversion ({file_ext})"
            
            if not svg_path:
                return {'success': False, 'error': 'Failed to process image'}
            
            # Generate embroidery pattern
            pattern_info = self._generate_embroidery_pattern(svg_path)
            
            if not pattern_info:
                return {'success': False, 'error': 'Failed to generate embroidery pattern'}
            
            # Generate DST file (internal use only)
            dst_file = self._generate_dst_file(pattern_info)
            
            # Generate preview image
            preview_path = self._generate_preview(pattern_info)
            
            if not preview_path:
                return {'success': False, 'error': 'Failed to generate preview'}
            
            # Clean up temporary SVG if it was created from raster
            if file_ext != '.svg' and os.path.exists(svg_path):
                os.remove(svg_path)
            
            return {
                'success': True,
                'preview_image': os.path.basename(preview_path),
                'info': {
                    'processing_type': processing_type,
                    'stitch_count': pattern_info['stitch_count'],
                    'color_count': pattern_info['color_count'],
                    'pattern_size': pattern_info['size']
                }
            }
            
        except Exception as e:
            logging.error(f"Processing error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _vectorize_image(self, image_path):
        """Convert raster image to SVG using advanced edge detection and contour analysis"""
        try:
            import cv2
            import numpy as np
            from skimage import measure
            
            # Open and process the image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if too large
                max_size = 800
                if img.width > max_size or img.height > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                # Convert PIL to OpenCV format
                img_array = np.array(img)
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # Advanced vectorization using contour detection
                svg_path = os.path.join(self.output_folder, f"vectorized_{os.path.basename(image_path)}.svg")
                self._create_advanced_svg_from_image(img_cv, svg_path)
                
                return svg_path
                
        except Exception as e:
            logging.error(f"Advanced vectorization error: {str(e)}")
            # Fallback to simple method
            return self._vectorize_image_simple(image_path)
    
    def _vectorize_image_simple(self, image_path):
        """Simple fallback vectorization method"""
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                max_size = 800
                if img.width > max_size or img.height > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                svg_path = os.path.join(self.output_folder, f"vectorized_{os.path.basename(image_path)}.svg")
                self._create_svg_from_image(img, svg_path)
                
                return svg_path
                
        except Exception as e:
            logging.error(f"Simple vectorization error: {str(e)}")
            return None
    
    def _create_advanced_svg_from_image(self, img_cv, svg_path):
        """Create SVG from image using advanced contour detection and color segmentation"""
        try:
            import cv2
            import numpy as np
            from sklearn.cluster import KMeans
            
            height, width = img_cv.shape[:2]
            
            # Color-based segmentation for better embroidery pattern detection
            # Reshape image to be a list of pixels
            img_data = img_cv.reshape((-1, 3))
            img_data = np.float32(img_data)
            
            # Apply K-means clustering to find dominant colors
            k = min(8, len(np.unique(img_data.reshape(-1, 3), axis=0)))  # Max 8 colors
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(img_data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Convert back to uint8 and reshape
            centers = np.uint8(centers)
            segmented_data = centers[labels.flatten()]
            segmented_img = segmented_data.reshape(img_cv.shape)
            
            # Create SVG content
            svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
'''
            
            # Process each color segment
            for i, center in enumerate(centers):
                # Create mask for this color
                color_mask = np.zeros(height * width, dtype=np.uint8)
                color_mask[labels.flatten() == i] = 255
                color_mask = color_mask.reshape((height, width))
                
                # Find contours for this color
                contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Convert color to hex
                color_hex = f"#{center[2]:02x}{center[1]:02x}{center[0]:02x}"
                
                # Process significant contours
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 200:  # Filter small areas
                        # Simplify contour
                        epsilon = 0.005 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        
                        # Create path from contour
                        if len(approx) >= 3:
                            path_data = "M"
                            for point in approx:
                                x, y = point[0]
                                path_data += f" {x},{y}"
                            path_data += " Z"
                            
                            svg_content += f'<path d="{path_data}" fill="{color_hex}" stroke="none"/>\n'
            
            svg_content += '</svg>'
            
            # Save SVG
            with open(svg_path, 'w', encoding='utf-8') as f:
                f.write(svg_content)
            
            logging.info(f"Advanced color-segmented SVG created: {svg_path}")
            
        except Exception as e:
            logging.error(f"Advanced SVG creation error: {str(e)}")
            # Fallback to simple contour method
            try:
                import cv2
                import numpy as np
                
                # Convert to grayscale
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                
                # Apply Gaussian blur to reduce noise
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                
                # Edge detection using adaptive threshold
                edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 11, 2)
                
                # Find contours
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                height, width = gray.shape
                
                # Create SVG content
                svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
'''
                
                # Process contours and create paths
                for i, contour in enumerate(contours):
                    if cv2.contourArea(contour) > 100:  # Filter small contours
                        # Approximate contour to reduce complexity
                        epsilon = 0.02 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        
                        # Get dominant color in contour area
                        mask = np.zeros(gray.shape, np.uint8)
                        cv2.fillPoly(mask, [contour], 255)
                        mean_color = cv2.mean(img_cv, mask=mask)
                        color = f"#{int(mean_color[2]):02x}{int(mean_color[1]):02x}{int(mean_color[0]):02x}"
                        
                        # Create path from contour
                        path_data = "M"
                        for point in approx:
                            x, y = point[0]
                            path_data += f" {x},{y}"
                        path_data += " Z"
                        
                        svg_content += f'<path d="{path_data}" fill="{color}" stroke="none"/>\n'
                
                svg_content += '</svg>'
                
                # Save SVG
                with open(svg_path, 'w', encoding='utf-8') as f:
                    f.write(svg_content)
                
                logging.info(f"Fallback contour SVG created: {svg_path}")
                
            except Exception as e2:
                logging.error(f"Fallback SVG creation error: {str(e2)}")
                # Last resort: simple method
                img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                self._create_svg_from_image(img_pil, svg_path)
    
    def _create_svg_from_image(self, img, svg_path):
        """Create SVG from image using color quantization"""
        # Quantize colors to reduce complexity
        quantized = img.quantize(colors=8, method=Image.Quantize.MEDIANCUT)
        quantized = quantized.convert('RGB')
        
        width, height = quantized.size
        
        # Create SVG content
        svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
'''
        
        # Process image in blocks to create shapes
        block_size = 20
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                # Get dominant color in block
                block_width = min(block_size, width - x)
                block_height = min(block_size, height - y)
                
                block = quantized.crop((x, y, x + block_width, y + block_height))
                colors = block.getcolors()
                
                if colors:
                    # Get most common color
                    dominant_color = max(colors, key=lambda c: c[0])[1]
                    r, g, b = dominant_color
                    
                    # Skip very light colors (assume background)
                    if r + g + b < 600:  # Threshold for non-background
                        svg_content += f'<rect x="{x}" y="{y}" width="{block_width}" height="{block_height}" fill="rgb({r},{g},{b})" />\n'
        
        svg_content += '</svg>'
        
        with open(svg_path, 'w') as f:
            f.write(svg_content)
    
    def _parse_svg_dimension(self, dimension_str):
        """Parse SVG dimension string (e.g., '100mm', '50px', '200') to float"""
        if not dimension_str:
            return 100.0
        
        # Remove common units and convert to float
        dimension_str = str(dimension_str).strip()
        
        # Handle common units
        units = ['px', 'pt', 'pc', 'mm', 'cm', 'in', '%']
        for unit in units:
            if dimension_str.endswith(unit):
                try:
                    value = float(dimension_str[:-len(unit)])
                    # Convert to pixels (approximate)
                    if unit == 'mm':
                        return value * 3.78  # 1mm ≈ 3.78px
                    elif unit == 'cm':
                        return value * 37.8  # 1cm ≈ 37.8px
                    elif unit == 'in':
                        return value * 96    # 1in = 96px
                    elif unit == 'pt':
                        return value * 1.33  # 1pt ≈ 1.33px
                    elif unit == 'pc':
                        return value * 16    # 1pc = 16px
                    elif unit == '%':
                        return value * 2     # Rough conversion
                    else:
                        return value  # px or unitless
                except ValueError:
                    return 100.0
        
        # Try to parse as plain number
        try:
            return float(dimension_str)
        except ValueError:
            return 100.0
    
    def _clean_svg_file(self, svg_path):
        """Clean and optimize SVG file to remove artifacts"""
        try:
            # Read SVG content with multiple encoding attempts
            svg_content = None
            for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                try:
                    with open(svg_path, 'r', encoding=encoding) as f:
                        svg_content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if svg_content is None:
                logging.error("Could not decode SVG file with any encoding")
                return svg_path
            
            # Fix XML structure issues
            svg_content = self._fix_xml_structure(svg_content)
            
            # Remove common problematic elements
            svg_content = self._remove_svg_artifacts(svg_content)
            
            # Create cleaned file
            cleaned_path = os.path.join(self.output_folder, f"cleaned_{os.path.basename(svg_path)}")
            with open(cleaned_path, 'w', encoding='utf-8') as f:
                f.write(svg_content)
            
            return cleaned_path
            
        except Exception as e:
            logging.error(f"SVG cleaning error: {str(e)}")
            return svg_path  # Return original if cleaning fails
    
    def _fix_xml_structure(self, svg_content):
        """Fix common XML structure issues in SVG files"""
        import re
        
        # Ensure proper XML declaration
        if not svg_content.strip().startswith('<?xml'):
            svg_content = '<?xml version="1.0" encoding="UTF-8"?>\n' + svg_content
        
        # Fix invalid characters (keep more characters for international content)
        svg_content = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF]', '', svg_content)
        
        # Ensure proper namespace declarations
        if '<svg' in svg_content and 'xmlns=' not in svg_content:
            svg_content = svg_content.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"')
        
        # Fix common SVG structure issues without breaking valid XML
        # Remove duplicate closing tags
        svg_content = re.sub(r'</svg>\s*</svg>', '</svg>', svg_content)
        
        return svg_content
    
    def _remove_svg_artifacts(self, svg_content):
        """Remove common SVG artifacts and problematic elements"""
        import re
        
        # Remove comments
        svg_content = re.sub(r'<!--.*?-->', '', svg_content, flags=re.DOTALL)
        
        # Remove metadata and sodipodi elements
        svg_content = re.sub(r'<metadata.*?</metadata>', '', svg_content, flags=re.DOTALL)
        svg_content = re.sub(r'<sodipodi:.*?>', '', svg_content)
        svg_content = re.sub(r'</sodipodi:.*?>', '', svg_content)
        
        # Remove inkscape-specific attributes
        svg_content = re.sub(r'inkscape:[^=]*="[^"]*"', '', svg_content)
        svg_content = re.sub(r'sodipodi:[^=]*="[^"]*"', '', svg_content)
        
        # Remove very small elements (artifacts)
        svg_content = re.sub(r'<rect[^>]*width="[0-9]*\.?[0-9]*"[^>]*height="[0-9]*\.?[0-9]*"[^>]*>', 
                            lambda m: '' if self._is_tiny_element(m.group()) else m.group(), svg_content)
        
        # Remove duplicate or overlapping elements
        svg_content = self._remove_duplicate_elements(svg_content)
        
        # Clean up whitespace
        svg_content = re.sub(r'\s+', ' ', svg_content)
        svg_content = re.sub(r'>\s+<', '><', svg_content)
        
        return svg_content
    
    def _is_tiny_element(self, element_str):
        """Check if element is too small to be meaningful"""
        import re
        
        width_match = re.search(r'width="([0-9]*\.?[0-9]*)"', element_str)
        height_match = re.search(r'height="([0-9]*\.?[0-9]*)"', element_str)
        
        if width_match and height_match:
            try:
                width = float(width_match.group(1))
                height = float(height_match.group(1))
                return width < 1 or height < 1
            except:
                return False
        return False
    
    def _remove_duplicate_elements(self, svg_content):
        """Remove duplicate or nearly identical elements"""
        # Simple deduplication - could be enhanced
        lines = svg_content.split('\n')
        seen = set()
        cleaned_lines = []
        
        for line in lines:
            line_clean = line.strip()
            if line_clean not in seen:
                seen.add(line_clean)
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _generate_embroidery_pattern(self, svg_path):
        """Generate embroidery pattern from SVG with cleanup and optimization"""
        try:
            # Try to parse SVG directly first
            try:
                tree = ET.parse(svg_path)
                root = tree.getroot()
                logging.info(f"Successfully parsed original SVG: {svg_path}")
            except ET.ParseError as e:
                logging.warning(f"Original SVG parse failed: {str(e)}, trying to clean...")
                # Clean and optimize SVG first
                cleaned_svg_path = self._clean_svg_file(svg_path)
                tree = ET.parse(cleaned_svg_path)
                root = tree.getroot()
                logging.info(f"Successfully parsed cleaned SVG: {cleaned_svg_path}")
                
                # Clean up temporary file
                if cleaned_svg_path != svg_path:
                    os.remove(cleaned_svg_path)
            
            # Extract SVG dimensions with unit conversion
            width = self._parse_svg_dimension(root.get('width', '100'))
            height = self._parse_svg_dimension(root.get('height', '100'))
            
            # Extract viewBox if available for better scaling
            viewbox = root.get('viewBox')
            if viewbox:
                try:
                    vb_values = viewbox.split()
                    if len(vb_values) == 4:
                        width = float(vb_values[2])
                        height = float(vb_values[3])
                except:
                    pass
            
            # Extract shapes and convert to stitch patterns
            shapes = []
            colors_used = set()
            
            # Process different SVG elements
            for elem in root.iter():
                tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                
                if tag_name == 'rect':
                    shape_data = self._process_rect(elem)
                    if shape_data:
                        shapes.append(shape_data)
                        colors_used.add(shape_data['color'])
                elif tag_name == 'circle':
                    shape_data = self._process_circle(elem)
                    if shape_data:
                        shapes.append(shape_data)
                        colors_used.add(shape_data['color'])
                elif tag_name == 'ellipse':
                    shape_data = self._process_ellipse(elem)
                    if shape_data:
                        shapes.append(shape_data)
                        colors_used.add(shape_data['color'])
                elif tag_name == 'path':
                    shape_data = self._process_path(elem)
                    if shape_data:
                        shapes.append(shape_data)
                        colors_used.add(shape_data['color'])
                elif tag_name == 'polygon':
                    shape_data = self._process_polygon(elem)
                    if shape_data:
                        shapes.append(shape_data)
                        colors_used.add(shape_data['color'])
                elif tag_name == 'line':
                    shape_data = self._process_line(elem)
                    if shape_data:
                        shapes.append(shape_data)
                        colors_used.add(shape_data['color'])
            
            # Calculate total stitches
            total_stitches = sum(len(shape['stitches']) for shape in shapes)
            
            logging.info(f"Generated embroidery pattern with {len(shapes)} shapes and {total_stitches} stitches")
            
            return {
                'shapes': shapes,
                'size': (width, height),
                'colors_used': list(colors_used),
                'stitch_count': total_stitches,
                'color_count': len(colors_used)
            }
            
        except Exception as e:
            logging.error(f"Pattern generation error: {str(e)}")
            import traceback
            logging.error(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def _process_rect(self, rect):
        """Process rectangle element"""
        try:
            x = float(rect.get('x', 0))
            y = float(rect.get('y', 0))
            width = float(rect.get('width', 0))
            height = float(rect.get('height', 0))
            fill = rect.get('fill', '#000000')
            
            # Generate stitches for rectangle outline and fill
            stitches = []
            
            # Outline stitches
            step = 5  # Stitch spacing
            
            # Top edge
            for i in range(0, int(width), step):
                stitches.append((x + i, y))
            
            # Right edge
            for i in range(0, int(height), step):
                stitches.append((x + width, y + i))
            
            # Bottom edge
            for i in range(int(width), 0, -step):
                stitches.append((x + i, y + height))
            
            # Left edge
            for i in range(int(height), 0, -step):
                stitches.append((x, y + i))
            
            # Fill stitches (simple back-and-forth pattern)
            if width > 10 and height > 10:
                fill_step = 8
                for row in range(fill_step, int(height), fill_step):
                    if row // fill_step % 2 == 0:
                        # Left to right
                        for col in range(fill_step, int(width), fill_step):
                            stitches.append((x + col, y + row))
                    else:
                        # Right to left
                        for col in range(int(width) - fill_step, fill_step, -fill_step):
                            stitches.append((x + col, y + row))
            
            return {
                'type': 'rect',
                'stitches': stitches,
                'color': fill
            }
            
        except Exception as e:
            logging.error(f"Rectangle processing error: {str(e)}")
            return None
    
    def _process_circle(self, circle):
        """Process circle element"""
        try:
            cx = float(circle.get('cx', 0))
            cy = float(circle.get('cy', 0))
            r = float(circle.get('r', 0))
            fill = circle.get('fill', '#000000')
            
            stitches = []
            
            # Generate circular stitches
            step = 0.2  # Angle step
            for angle in range(0, int(2 * math.pi / step)):
                x = cx + r * math.cos(angle * step)
                y = cy + r * math.sin(angle * step)
                stitches.append((x, y))
            
            # Fill with concentric circles
            if r > 5:
                for radius in range(5, int(r), 5):
                    for angle in range(0, int(2 * math.pi / step), 2):
                        x = cx + radius * math.cos(angle * step)
                        y = cy + radius * math.sin(angle * step)
                        stitches.append((x, y))
            
            return {
                'type': 'circle',
                'stitches': stitches,
                'color': fill
            }
            
        except Exception as e:
            logging.error(f"Circle processing error: {str(e)}")
            return None
    
    def _process_path(self, path):
        """Process path element (simplified)"""
        try:
            d = path.get('d', '')
            fill = path.get('fill', '#000000')
            
            # Simple path processing - just extract coordinates
            stitches = []
            
            # Basic parsing of path commands
            import re
            coords = re.findall(r'-?\d+\.?\d*', d)
            
            for i in range(0, len(coords) - 1, 2):
                try:
                    x = float(coords[i])
                    y = float(coords[i + 1])
                    stitches.append((x, y))
                except (ValueError, IndexError):
                    continue
            
            return {
                'type': 'path',
                'stitches': stitches,
                'color': fill
            }
            
        except Exception as e:
            logging.error(f"Path processing error: {str(e)}")
            return None
    
    def _process_ellipse(self, ellipse):
        """Process ellipse element"""
        try:
            cx = float(ellipse.get('cx', 0))
            cy = float(ellipse.get('cy', 0))
            rx = float(ellipse.get('rx', 0))
            ry = float(ellipse.get('ry', 0))
            fill = ellipse.get('fill', '#000000')
            
            stitches = []
            
            # Generate elliptical stitches
            step = 0.2
            for angle in range(0, int(2 * math.pi / step)):
                x = cx + rx * math.cos(angle * step)
                y = cy + ry * math.sin(angle * step)
                stitches.append((x, y))
            
            return {
                'type': 'ellipse',
                'stitches': stitches,
                'color': fill
            }
            
        except Exception as e:
            logging.error(f"Ellipse processing error: {str(e)}")
            return None
    
    def _process_polygon(self, polygon):
        """Process polygon element"""
        try:
            points = polygon.get('points', '')
            fill = polygon.get('fill', '#000000')
            
            stitches = []
            
            # Parse points
            coords = points.replace(',', ' ').split()
            for i in range(0, len(coords) - 1, 2):
                try:
                    x = float(coords[i])
                    y = float(coords[i + 1])
                    stitches.append((x, y))
                except (ValueError, IndexError):
                    continue
            
            return {
                'type': 'polygon',
                'stitches': stitches,
                'color': fill
            }
            
        except Exception as e:
            logging.error(f"Polygon processing error: {str(e)}")
            return None
    
    def _process_line(self, line):
        """Process line element"""
        try:
            x1 = float(line.get('x1', 0))
            y1 = float(line.get('y1', 0))
            x2 = float(line.get('x2', 0))
            y2 = float(line.get('y2', 0))
            stroke = line.get('stroke', '#000000')
            
            stitches = [(x1, y1), (x2, y2)]
            
            return {
                'type': 'line',
                'stitches': stitches,
                'color': stroke
            }
            
        except Exception as e:
            logging.error(f"Line processing error: {str(e)}")
            return None
    
    def _generate_dst_file(self, pattern_info):
        """Generate Tajima-compatible DST embroidery file"""
        try:
            # Create embroidery pattern
            pattern = pyembroidery.EmbPattern()
            
            # Set pattern metadata for Tajima compatibility
            pattern.extras["label"] = f"Pattern_{random.randint(1000, 9999)}"
            pattern.extras["author"] = "EmbroideryGenerator"
            
            # Calculate design bounds
            all_stitches = []
            for shape in pattern_info['shapes']:
                all_stitches.extend(shape['stitches'])
            
            if all_stitches:
                min_x = min(stitch[0] for stitch in all_stitches)
                max_x = max(stitch[0] for stitch in all_stitches)
                min_y = min(stitch[1] for stitch in all_stitches)
                max_y = max(stitch[1] for stitch in all_stitches)
                
                # Set bounds
                pattern.extends = (min_x * 10, min_y * 10, max_x * 10, max_y * 10)
            
            # Add stitches by color with proper Tajima formatting
            for shape_idx, shape in enumerate(pattern_info['shapes']):
                color = shape['color']
                stitches = shape['stitches']
                
                if not stitches:
                    continue
                
                # Convert color to thread color
                if color.startswith('#'):
                    color_int = int(color[1:], 16)
                else:
                    color_int = 0x000000
                
                # Color change for all shapes
                pattern.color_change(color_int)
                
                # Move to first stitch position
                if stitches:
                    first_x, first_y = stitches[0]
                    pattern.move_abs(first_x * 10, first_y * 10)
                
                # Add stitches with proper spacing for Tajima machines
                for i, (x, y) in enumerate(stitches):
                    scaled_x = x * 10  # Scale to 0.1mm units
                    scaled_y = y * 10
                    
                    # Add regular stitch
                    pattern.stitch_abs(scaled_x, scaled_y)
                    
                    # Add trim after each shape (except last)
                    if i == len(stitches) - 1 and shape_idx < len(pattern_info['shapes']) - 1:
                        pattern.trim()
            
            # End pattern properly for Tajima
            pattern.end()
            
            # Save DST file with proper settings
            dst_path = os.path.join(self.output_folder, f"pattern_{random.randint(1000, 9999)}.dst")
            
            # Write with specific settings for Tajima compatibility
            pyembroidery.write_dst(pattern, dst_path, {
                "write_file_format": "dst",
                "encode_header": True,
                "encode_metadata": True,
                "max_stitch_length": 127,  # Tajima limit
                "max_jump_length": 127,    # Tajima limit
                "full_jump": True,
                "sequin_contingency": True,
                "trim_at": 3,  # Auto-trim after 3 jumps
                "tie_on": False,
                "tie_off": False
            })
            
            logging.info(f"Tajima-compatible DST file generated: {dst_path}")
            return dst_path
            
        except Exception as e:
            logging.error(f"DST generation error: {str(e)}")
            # Fallback to basic DST generation
            try:
                pattern = pyembroidery.EmbPattern()
                for shape in pattern_info['shapes']:
                    for stitch in shape['stitches']:
                        x, y = stitch
                        pattern.stitch_abs(x * 10, y * 10)
                pattern.end()
                
                dst_path = os.path.join(self.output_folder, f"pattern_{random.randint(1000, 9999)}.dst")
                pyembroidery.write_dst(pattern, dst_path)
                
                logging.info(f"Basic DST file generated: {dst_path}")
                return dst_path
            except Exception as e2:
                logging.error(f"Fallback DST generation failed: {str(e2)}")
                return None
    
    def _generate_preview(self, pattern_info):
        """Generate high-quality machine embroidery preview"""
        try:
            return self._generate_high_quality_preview(pattern_info)
        except Exception as e:
            logging.error(f"High-quality preview generation error: {str(e)}")
            # Fallback to simple method
            return self._generate_simple_preview(pattern_info)
    
    def _generate_high_quality_preview(self, pattern_info):
        """Generate high-quality machine embroidery preview using PIL and advanced techniques"""
        try:
            import cv2
            import numpy as np
            
            width, height = pattern_info['size']
            
            # Create high-resolution canvas (4x for better quality)
            scale_factor = 4
            canvas_width = int(width * scale_factor)
            canvas_height = int(height * scale_factor)
            padding = 100
            
            # Create base image with fabric background
            img_width = canvas_width + 2 * padding
            img_height = canvas_height + 2 * padding
            
            # Create high-quality fabric background
            fabric_img = self._create_fabric_background(img_width, img_height)
            
            # Convert to numpy array for OpenCV processing
            fabric_array = np.array(fabric_img)
            
            # Process each shape with high-quality rendering
            for shape in pattern_info['shapes']:
                color = shape['color']
                stitches = shape['stitches']
                shape_type = shape.get('type', 'unknown')
                
                if len(stitches) < 2:
                    continue
                
                # Scale stitches to high resolution
                scaled_stitches = [(int(s[0] * scale_factor + padding), 
                                  int(s[1] * scale_factor + padding)) for s in stitches]
                
                # Draw embroidery with realistic thread effects
                fabric_array = self._draw_high_quality_embroidery(
                    fabric_array, scaled_stitches, color, shape_type
                )
            
            # Convert back to PIL Image
            result_img = Image.fromarray(fabric_array)
            
            # Add embroidery hoop effect
            result_img = self._add_embroidery_hoop(result_img, img_width, img_height)
            
            # Add professional information overlay
            result_img = self._add_embroidery_info_overlay(result_img, pattern_info)
            
            # Resize to final output size (downsample for anti-aliasing)
            final_width = int(img_width * 0.8)
            final_height = int(img_height * 0.8)
            result_img = result_img.resize((final_width, final_height), Image.LANCZOS)
            
            # Save high-quality preview
            preview_path = os.path.join(self.output_folder, f"preview_{random.randint(1000, 9999)}.png")
            result_img.save(preview_path, 'PNG', quality=95, optimize=True)
            
            logging.info(f"High-quality machine embroidery preview generated: {preview_path}")
            return preview_path
            
        except Exception as e:
            logging.error(f"High-quality preview error: {str(e)}")
            raise e
    
    def _create_fabric_background(self, width, height):
        """Create realistic fabric background texture"""
        try:
            import cv2
            import numpy as np
            
            # Create base fabric color
            base_color = (248, 248, 240)  # Light beige fabric
            fabric_img = np.full((height, width, 3), base_color, dtype=np.uint8)
            
            # Add fabric weave pattern
            weave_size = 4
            for y in range(0, height, weave_size):
                for x in range(0, width, weave_size):
                    # Add subtle weave pattern
                    if (x // weave_size + y // weave_size) % 2 == 0:
                        fabric_img[y:y+weave_size, x:x+weave_size] = [245, 245, 235]
                    else:
                        fabric_img[y:y+weave_size, x:x+weave_size] = [250, 250, 245]
            
            # Add noise for texture
            noise = np.random.randint(-5, 5, (height, width, 3))
            fabric_img = np.clip(fabric_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Apply slight Gaussian blur for softness
            fabric_img = cv2.GaussianBlur(fabric_img, (3, 3), 0)
            
            return Image.fromarray(fabric_img)
            
        except Exception as e:
            logging.error(f"Fabric background creation error: {str(e)}")
            # Fallback to simple background
            return Image.new('RGB', (width, height), '#f8f8f0')
    
    def _draw_high_quality_embroidery(self, fabric_array, stitches, color, shape_type):
        """Draw high-quality embroidery with realistic thread effects"""
        try:
            import cv2
            import numpy as np
            
            # Convert hex color to BGR
            if color.startswith('#'):
                hex_color = color[1:]
                r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
                thread_color = (b, g, r)  # OpenCV uses BGR
            else:
                thread_color = (100, 100, 100)  # Default gray
            
            if shape_type == 'rect' or shape_type == 'polygon':
                return self._draw_satin_stitch_fill(fabric_array, stitches, thread_color)
            elif shape_type == 'circle' or shape_type == 'ellipse':
                return self._draw_radial_stitch_fill(fabric_array, stitches, thread_color)
            else:
                return self._draw_running_stitch_fill(fabric_array, stitches, thread_color)
            
        except Exception as e:
            logging.error(f"High-quality embroidery drawing error: {str(e)}")
            return fabric_array
    
    def _draw_satin_stitch_fill(self, fabric_array, stitches, thread_color):
        """Draw satin stitch fill pattern"""
        try:
            import cv2
            import numpy as np
            
            if len(stitches) < 3:
                return fabric_array
            
            # Get bounding box
            x_coords = [s[0] for s in stitches]
            y_coords = [s[1] for s in stitches]
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            # Draw satin fill lines
            line_spacing = 3
            for y in range(min_y, max_y, line_spacing):
                # Add thread texture variation
                thread_var = np.random.randint(-15, 15, 3)
                varied_color = tuple(np.clip(np.array(thread_color) + thread_var, 0, 255))
                
                # Draw main stitch line
                cv2.line(fabric_array, (min_x, y), (max_x, y), varied_color, 2)
                
                # Add highlight thread effect
                if y % 6 == 0:
                    highlight_color = tuple(np.clip(np.array(thread_color) + 30, 0, 255))
                    cv2.line(fabric_array, (min_x, y), (max_x, y), highlight_color, 1)
            
            # Draw outline
            outline_color = tuple(np.clip(np.array(thread_color) - 40, 0, 255))
            pts = np.array(stitches, np.int32)
            cv2.polylines(fabric_array, [pts], True, outline_color, 2)
            
            return fabric_array
            
        except Exception as e:
            logging.error(f"Satin stitch drawing error: {str(e)}")
            return fabric_array
    
    def _draw_radial_stitch_fill(self, fabric_array, stitches, thread_color):
        """Draw radial stitch fill for circular shapes"""
        try:
            import cv2
            import numpy as np
            
            if len(stitches) < 3:
                return fabric_array
            
            # Calculate center
            x_coords = [s[0] for s in stitches]
            y_coords = [s[1] for s in stitches]
            center_x = int(sum(x_coords) / len(x_coords))
            center_y = int(sum(y_coords) / len(y_coords))
            
            # Calculate radius
            distances = [((x - center_x)**2 + (y - center_y)**2)**0.5 for x, y in zip(x_coords, y_coords)]
            radius = int(max(distances))
            
            # Draw concentric circles
            for r in range(5, radius, 8):
                thread_var = np.random.randint(-10, 10, 3)
                varied_color = tuple(np.clip(np.array(thread_color) + thread_var, 0, 255))
                cv2.circle(fabric_array, (center_x, center_y), r, varied_color, 2)
            
            # Fill center
            cv2.circle(fabric_array, (center_x, center_y), 4, thread_color, -1)
            
            return fabric_array
            
        except Exception as e:
            logging.error(f"Radial stitch drawing error: {str(e)}")
            return fabric_array
    
    def _draw_running_stitch_fill(self, fabric_array, stitches, thread_color):
        """Draw running stitch fill pattern"""
        try:
            import cv2
            import numpy as np
            
            if len(stitches) < 2:
                return fabric_array
            
            # Draw running stitches
            for i in range(len(stitches) - 1):
                start_point = stitches[i]
                end_point = stitches[i + 1]
                
                # Add thread texture variation
                thread_var = np.random.randint(-10, 10, 3)
                varied_color = tuple(np.clip(np.array(thread_color) + thread_var, 0, 255))
                
                cv2.line(fabric_array, start_point, end_point, varied_color, 2)
            
            return fabric_array
            
        except Exception as e:
            logging.error(f"Running stitch drawing error: {str(e)}")
            return fabric_array
    
    def _add_embroidery_hoop(self, img, width, height):
        """Add embroidery hoop frame effect"""
        try:
            # Create hoop overlay
            hoop_overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            hoop_draw = ImageDraw.Draw(hoop_overlay)
            
            # Draw outer hoop
            hoop_padding = 40
            hoop_draw.ellipse([hoop_padding, hoop_padding, width - hoop_padding, height - hoop_padding],
                             outline=(139, 69, 19, 200), width=12)
            
            # Draw inner hoop
            inner_padding = hoop_padding + 8
            hoop_draw.ellipse([inner_padding, inner_padding, width - inner_padding, height - inner_padding],
                             outline=(160, 82, 45, 150), width=8)
            
            # Composite hoop onto image
            img = Image.alpha_composite(img.convert('RGBA'), hoop_overlay)
            return img.convert('RGB')
            
        except Exception as e:
            logging.error(f"Hoop effect error: {str(e)}")
            return img
    
    def _add_embroidery_info_overlay(self, img, pattern_info):
        """Add professional embroidery information overlay"""
        try:
            # Create info overlay
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            
            # Draw info box
            info_box = [20, 20, 300, 120]
            overlay_draw.rounded_rectangle(info_box, radius=10, fill=(255, 255, 255, 200), outline=(0, 0, 0, 100))
            
            # Add text information
            info_text = f"Machine Embroidery Preview\n"
            info_text += f"Stitches: {pattern_info['stitch_count']}\n"
            info_text += f"Colors: {pattern_info['color_count']}\n"
            info_text += f"Size: {int(pattern_info['size'][0])}x{int(pattern_info['size'][1])}"
            
            overlay_draw.text((30, 30), info_text, fill=(0, 0, 0, 255))
            
            # Add color legend
            legend_x = 320
            legend_y = 30
            for i, color in enumerate(pattern_info['colors_used'][:5]):
                color_box = [legend_x, legend_y + i * 25, legend_x + 20, legend_y + i * 25 + 20]
                overlay_draw.rectangle(color_box, fill=color, outline=(0, 0, 0, 255))
                overlay_draw.text((legend_x + 25, legend_y + i * 25 + 5), f"Thread {i+1}", fill=(0, 0, 0, 255))
            
            # Composite overlay onto image
            img = Image.alpha_composite(img.convert('RGBA'), overlay)
            return img.convert('RGB')
            
        except Exception as e:
            logging.error(f"Info overlay error: {str(e)}")
            return img
    
    def _generate_simple_preview(self, pattern_info):
        """Simple fallback preview generation"""
        try:
            width, height = pattern_info['size']
            
            # Create image with padding
            padding = 50
            img_width = int(width) + 2 * padding
            img_height = int(height) + 2 * padding
            
            # Create image with fabric background
            img = Image.new('RGB', (img_width, img_height), '#f8f8f0')
            draw = ImageDraw.Draw(img)
            
            # Add subtle fabric texture
            self._add_fabric_texture(draw, img_width, img_height)
            
            # Draw realistic embroidery for each shape
            for shape in pattern_info['shapes']:
                color = shape['color']
                stitches = shape['stitches']
                shape_type = shape.get('type', 'unknown')
                
                if len(stitches) < 2:
                    continue
                
                # Draw based on shape type
                if shape_type == 'rect':
                    self._draw_filled_rectangle(draw, stitches, color, padding)
                elif shape_type == 'circle':
                    self._draw_filled_circle(draw, stitches, color, padding)
                else:
                    self._draw_filled_shape(draw, stitches, color, padding)
            
            # Draw color blocks (thread legend)
            self._draw_color_blocks(draw, pattern_info['colors_used'], img_width, img_height)
            
            # Add stitch count information
            self._add_stitch_info(draw, pattern_info, img_width, img_height)
            
            # Save preview image
            preview_path = os.path.join(self.output_folder, f"preview_{random.randint(1000, 9999)}.png")
            img.save(preview_path, 'PNG')
            
            logging.info(f"Simple machine embroidery preview generated: {preview_path}")
            return preview_path
            
        except Exception as e:
            logging.error(f"Simple preview generation error: {str(e)}")
            return None
    
    def _add_matplotlib_fabric_texture(self, ax, width, height):
        """Add fabric texture using matplotlib"""
        import numpy as np
        
        # Add small random dots to simulate fabric weave
        x_dots = np.random.uniform(0, width, int(width * height // 500))
        y_dots = np.random.uniform(0, height, int(width * height // 500))
        ax.scatter(x_dots, y_dots, s=0.5, c='#e8e8e0', alpha=0.3)
    
    def _draw_matplotlib_filled_rectangle(self, ax, stitches, color):
        """Draw filled rectangle with satin stitch pattern using matplotlib"""
        import matplotlib.patches as patches
        import numpy as np
        
        if len(stitches) < 4:
            return
        
        # Get rectangle bounds
        x_coords = [s[0] for s in stitches]
        y_coords = [s[1] for s in stitches]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Draw satin fill lines
        line_spacing = 3
        for y in np.arange(min_y, max_y, line_spacing):
            # Add slight variation to simulate thread texture
            line_color = self._add_thread_texture(color)
            ax.plot([min_x, max_x], [y, y], color=line_color, linewidth=2, alpha=0.8)
            
            # Add highlight lines
            if int(y) % 6 == 0:
                highlight_color = self._lighten_color(color, 0.3)
                ax.plot([min_x, max_x], [y, y], color=highlight_color, linewidth=1, alpha=0.6)
        
        # Draw outline
        outline_color = self._darken_color(color, 0.3)
        rect = patches.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                               linewidth=2, edgecolor=outline_color, facecolor='none')
        ax.add_patch(rect)
    
    def _draw_matplotlib_filled_circle(self, ax, stitches, color):
        """Draw filled circle with radial stitch pattern using matplotlib"""
        import matplotlib.patches as patches
        import numpy as np
        
        if len(stitches) < 3:
            return
        
        # Calculate center and radius
        x_coords = [s[0] for s in stitches]
        y_coords = [s[1] for s in stitches]
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        
        # Estimate radius
        distances = [((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5 
                    for x, y in zip(x_coords, y_coords)]
        radius = max(distances) if distances else 10
        
        # Draw concentric circles
        for r in np.arange(2, radius, 4):
            circle_color = self._add_thread_texture(color)
            circle = patches.Circle((center_x, center_y), r, 
                                  linewidth=2, edgecolor=circle_color, 
                                  facecolor='none', alpha=0.8)
            ax.add_patch(circle)
        
        # Fill center
        center_circle = patches.Circle((center_x, center_y), 3, 
                                     facecolor=color, alpha=0.9)
        ax.add_patch(center_circle)
    
    def _draw_matplotlib_filled_shape(self, ax, stitches, color):
        """Draw filled shape with appropriate stitch pattern using matplotlib"""
        import matplotlib.patches as patches
        import numpy as np
        
        if len(stitches) < 3:
            return
        
        # Get shape bounds
        x_coords = [s[0] for s in stitches]
        y_coords = [s[1] for s in stitches]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Determine fill direction
        width = max_x - min_x
        height = max_y - min_y
        
        # Draw fill pattern
        line_spacing = 3
        if width > height:
            # Horizontal fill
            for y in np.arange(min_y, max_y, line_spacing):
                line_color = self._add_thread_texture(color)
                ax.plot([min_x, max_x], [y, y], color=line_color, linewidth=2, alpha=0.8)
                
                # Add occasional highlight
                if int(y) % 6 == 0:
                    highlight_color = self._lighten_color(color, 0.2)
                    ax.plot([min_x, max_x], [y, y], color=highlight_color, linewidth=1, alpha=0.6)
        else:
            # Vertical fill
            for x in np.arange(min_x, max_x, line_spacing):
                line_color = self._add_thread_texture(color)
                ax.plot([x, x], [min_y, max_y], color=line_color, linewidth=2, alpha=0.8)
                
                # Add occasional highlight
                if int(x) % 6 == 0:
                    highlight_color = self._lighten_color(color, 0.2)
                    ax.plot([x, x], [min_y, max_y], color=highlight_color, linewidth=1, alpha=0.6)
        
        # Draw outline
        outline_color = self._darken_color(color, 0.3)
        if len(stitches) > 2:
            shape_points = [(s[0], s[1]) for s in stitches]
            polygon = patches.Polygon(shape_points, linewidth=2, 
                                    edgecolor=outline_color, facecolor='none')
            ax.add_patch(polygon)
    
    def _add_matplotlib_hoop_effect(self, ax, width, height):
        """Add embroidery hoop border effect using matplotlib"""
        import matplotlib.patches as patches
        
        # Add hoop border
        hoop_padding = 20
        hoop = patches.Rectangle((-hoop_padding, -hoop_padding), 
                               width + 2 * hoop_padding, height + 2 * hoop_padding,
                               linewidth=8, edgecolor='#8B4513', facecolor='none')
        ax.add_patch(hoop)
        
        # Add inner hoop
        inner_hoop = patches.Rectangle((-hoop_padding + 5, -hoop_padding + 5), 
                                     width + 2 * hoop_padding - 10, height + 2 * hoop_padding - 10,
                                     linewidth=4, edgecolor='#A0522D', facecolor='none')
        ax.add_patch(inner_hoop)
    
    def _add_matplotlib_color_legend(self, ax, colors_used):
        """Add color legend using matplotlib"""
        import matplotlib.patches as patches
        
        # Add color legend in top right
        legend_x = 0.85
        legend_y = 0.95
        
        for i, color in enumerate(colors_used[:5]):  # Limit to 5 colors
            rect = patches.Rectangle((legend_x, legend_y - i * 0.08), 0.05, 0.05,
                                   transform=ax.transAxes, facecolor=color, 
                                   edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            ax.text(legend_x + 0.07, legend_y - i * 0.08 + 0.025, f'Color {i+1}',
                   transform=ax.transAxes, fontsize=8, verticalalignment='center')
    
    def _draw_thread_path(self, draw, stitch_sequence, color_sequence, padding):
        """Draw realistic embroidery with satin fill and contour lines"""
        if len(stitch_sequence) < 2:
            return
        
        # Group stitches by color to create proper fill patterns
        color_groups = {}
        for i, (stitch, color) in enumerate(zip(stitch_sequence, color_sequence)):
            if color not in color_groups:
                color_groups[color] = []
            color_groups[color].append((stitch, i))
        
        # Draw each color group with proper embroidery technique
        for color, stitch_data in color_groups.items():
            stitches = [s[0] for s in stitch_data]
            
            # Create fill pattern for larger areas
            self._draw_satin_fill(draw, stitches, color, padding)
            
            # Draw contour lines
            self._draw_contour_lines(draw, stitches, color, padding)
    
    def _draw_satin_fill(self, draw, stitches, color, padding):
        """Draw satin stitch fill pattern like in machine embroidery"""
        if len(stitches) < 4:
            return
        
        # Calculate bounding box
        min_x = min(s[0] for s in stitches)
        max_x = max(s[0] for s in stitches)
        min_y = min(s[1] for s in stitches)
        max_y = max(s[1] for s in stitches)
        
        # Create satin fill with parallel lines
        line_spacing = 3  # Distance between satin lines
        
        # Determine fill direction (horizontal or vertical based on shape)
        width = max_x - min_x
        height = max_y - min_y
        
        if width > height:
            # Fill vertically
            for x in range(int(min_x), int(max_x), line_spacing):
                y_start = min_y
                y_end = max_y
                
                # Draw satin line with slight variation
                line_color = self._add_thread_texture(color)
                draw.line([(x + padding, y_start + padding), 
                          (x + padding, y_end + padding)], 
                         fill=line_color, width=2)
                
                # Add highlight line
                highlight_color = self._lighten_color(color, 0.2)
                draw.line([(x + padding - 1, y_start + padding), 
                          (x + padding - 1, y_end + padding)], 
                         fill=highlight_color, width=1)
        else:
            # Fill horizontally
            for y in range(int(min_y), int(max_y), line_spacing):
                x_start = min_x
                x_end = max_x
                
                # Draw satin line with slight variation
                line_color = self._add_thread_texture(color)
                draw.line([(x_start + padding, y + padding), 
                          (x_end + padding, y + padding)], 
                         fill=line_color, width=2)
                
                # Add highlight line
                highlight_color = self._lighten_color(color, 0.2)
                draw.line([(x_start + padding, y + padding - 1), 
                          (x_end + padding, y + padding - 1)], 
                         fill=highlight_color, width=1)
    
    def _draw_contour_lines(self, draw, stitches, color, padding):
        """Draw contour lines around shapes"""
        if len(stitches) < 2:
            return
        
        # Draw outline with darker color
        outline_color = self._darken_color(color, 0.4)
        
        for i in range(len(stitches) - 1):
            x1, y1 = stitches[i]
            x2, y2 = stitches[i + 1]
            
            # Draw contour line
            draw.line([(x1 + padding, y1 + padding), 
                      (x2 + padding, y2 + padding)], 
                     fill=outline_color, width=2)
        
        # Connect last to first for closed shapes
        if len(stitches) > 3:
            x1, y1 = stitches[-1]
            x2, y2 = stitches[0]
            draw.line([(x1 + padding, y1 + padding), 
                      (x2 + padding, y2 + padding)], 
                     fill=outline_color, width=2)
    
    def _add_thread_texture(self, color):
        """Add slight color variation to simulate thread texture"""
        try:
            # Remove # if present
            hex_color = color.lstrip('#')
            
            # Convert to RGB
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            
            # Add small random variation
            variation = random.randint(-10, 10)
            r = max(0, min(255, r + variation))
            g = max(0, min(255, g + variation))
            b = max(0, min(255, b + variation))
            
            return f'#{r:02x}{g:02x}{b:02x}'
        except:
            return color
    
    def _add_fabric_texture(self, draw, width, height):
        """Add subtle fabric texture to background"""
        # Add small random dots to simulate fabric weave
        for _ in range(width * height // 200):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            color = random.choice(['#f0f0e8', '#f5f5ed', '#eeeedc'])
            draw.point((x, y), fill=color)
    
    def _draw_filled_rectangle(self, draw, stitches, color, padding):
        """Draw filled rectangle with satin stitch pattern"""
        if len(stitches) < 4:
            return
        
        # Get rectangle bounds
        min_x = min(s[0] for s in stitches)
        max_x = max(s[0] for s in stitches)
        min_y = min(s[1] for s in stitches)
        max_y = max(s[1] for s in stitches)
        
        # Draw satin fill
        line_spacing = 2
        for y in range(int(min_y), int(max_y), line_spacing):
            line_color = self._add_thread_texture(color)
            draw.line([(min_x + padding, y + padding), 
                      (max_x + padding, y + padding)], 
                     fill=line_color, width=2)
            
            # Add thread shine effect
            if y % 4 == 0:
                highlight_color = self._lighten_color(color, 0.3)
                draw.line([(min_x + padding, y + padding), 
                          (max_x + padding, y + padding)], 
                         fill=highlight_color, width=1)
        
        # Draw outline
        outline_color = self._darken_color(color, 0.3)
        draw.rectangle([min_x + padding, min_y + padding, 
                       max_x + padding, max_y + padding], 
                      outline=outline_color, width=2)
    
    def _draw_filled_circle(self, draw, stitches, color, padding):
        """Draw filled circle with radial stitch pattern"""
        if len(stitches) < 3:
            return
        
        # Calculate center and radius
        center_x = sum(s[0] for s in stitches) / len(stitches)
        center_y = sum(s[1] for s in stitches) / len(stitches)
        
        # Estimate radius
        distances = [((s[0] - center_x) ** 2 + (s[1] - center_y) ** 2) ** 0.5 for s in stitches]
        radius = max(distances) if distances else 10
        
        # Draw concentric circles
        for r in range(2, int(radius), 3):
            circle_color = self._add_thread_texture(color)
            draw.ellipse([center_x - r + padding, center_y - r + padding,
                         center_x + r + padding, center_y + r + padding], 
                        outline=circle_color, width=2)
        
        # Fill center
        draw.ellipse([center_x - 2 + padding, center_y - 2 + padding,
                     center_x + 2 + padding, center_y + 2 + padding], 
                    fill=color)
    
    def _draw_filled_shape(self, draw, stitches, color, padding):
        """Draw filled shape with appropriate stitch pattern"""
        if len(stitches) < 3:
            return
        
        # Get shape bounds
        min_x = min(s[0] for s in stitches)
        max_x = max(s[0] for s in stitches)
        min_y = min(s[1] for s in stitches)
        max_y = max(s[1] for s in stitches)
        
        # Determine if shape is more horizontal or vertical
        width = max_x - min_x
        height = max_y - min_y
        
        # Draw fill pattern
        line_spacing = 3
        if width > height:
            # Horizontal fill
            for y in range(int(min_y), int(max_y), line_spacing):
                line_color = self._add_thread_texture(color)
                draw.line([(min_x + padding, y + padding), 
                          (max_x + padding, y + padding)], 
                         fill=line_color, width=2)
                
                # Add occasional highlight
                if y % 6 == 0:
                    highlight_color = self._lighten_color(color, 0.2)
                    draw.line([(min_x + padding, y + padding), 
                              (max_x + padding, y + padding)], 
                             fill=highlight_color, width=1)
        else:
            # Vertical fill
            for x in range(int(min_x), int(max_x), line_spacing):
                line_color = self._add_thread_texture(color)
                draw.line([(x + padding, min_y + padding), 
                          (x + padding, max_y + padding)], 
                         fill=line_color, width=2)
                
                # Add occasional highlight
                if x % 6 == 0:
                    highlight_color = self._lighten_color(color, 0.2)
                    draw.line([(x + padding, min_y + padding), 
                              (x + padding, max_y + padding)], 
                             fill=highlight_color, width=1)
        
        # Draw outline
        outline_color = self._darken_color(color, 0.3)
        adjusted_stitches = [(s[0] + padding, s[1] + padding) for s in stitches]
        if len(adjusted_stitches) > 2:
            for i in range(len(adjusted_stitches)):
                start = adjusted_stitches[i]
                end = adjusted_stitches[(i + 1) % len(adjusted_stitches)]
                draw.line([start, end], fill=outline_color, width=2)
    
    def _draw_color_blocks(self, draw, colors_used, img_width, img_height):
        """Draw color legend blocks like Inkstitch"""
        if not colors_used:
            return
        
        block_size = 20
        start_x = 10
        start_y = 10
        
        for i, color in enumerate(colors_used):
            x = start_x
            y = start_y + i * (block_size + 5)
            
            # Draw color block
            draw.rectangle([x, y, x + block_size, y + block_size], 
                         fill=color, outline='#000000', width=1)
            
            # Add thread number
            try:
                # Simple font rendering
                draw.text((x + block_size + 5, y + 2), f"Color {i+1}", fill='#000000')
            except:
                pass
    
    def _add_stitch_info(self, draw, pattern_info, img_width, img_height):
        """Add stitch count and size information"""
        try:
            info_text = f"Stitches: {pattern_info['stitch_count']}"
            size_text = f"Size: {pattern_info['size'][0]:.0f} x {pattern_info['size'][1]:.0f}"
            
            # Position at bottom right
            draw.text((img_width - 150, img_height - 40), info_text, fill='#000000')
            draw.text((img_width - 150, img_height - 20), size_text, fill='#000000')
        except:
            pass
    
    def _draw_thread_stitch(self, draw, x1, y1, x2, y2, color):
        """Draw a single thread stitch with realistic appearance"""
        # Main thread line
        draw.line([(x1, y1), (x2, y2)], fill=color, width=3)
        
        # Add thread highlights and shadows for 3D effect
        highlight_color = self._lighten_color(color, 0.3)
        shadow_color = self._darken_color(color, 0.3)
        
        # Offset for highlight/shadow
        dx = 1 if x2 > x1 else -1
        dy = 1 if y2 > y1 else -1
        
        # Highlight line
        draw.line([(x1 + dx, y1 + dy), (x2 + dx, y2 + dy)], fill=highlight_color, width=1)
        
        # Shadow line
        draw.line([(x1 - dx, y1 - dy), (x2 - dx, y2 - dy)], fill=shadow_color, width=1)
    
    def _draw_stitch_point(self, draw, x, y, color, point_type):
        """Draw stitch entry/exit point"""
        if point_type == 'entry':
            # Small dot for entry point
            draw.ellipse([x-1, y-1, x+1, y+1], fill=color)
        else:
            # Slightly larger dot for exit point
            draw.ellipse([x-2, y-2, x+2, y+2], fill=self._darken_color(color, 0.2))
    
    def _add_hoop_effect(self, draw, width, height, padding):
        """Add embroidery hoop border effect"""
        # Draw hoop outline
        hoop_color = '#8B4513'  # Brown wood color
        
        # Outer hoop
        draw.rectangle([padding//2, padding//2, width-padding//2, height-padding//2], 
                      outline=hoop_color, width=8)
        
        # Inner hoop (slightly smaller)
        draw.rectangle([padding//2 + 4, padding//2 + 4, width-padding//2 - 4, height-padding//2 - 4], 
                      outline=self._lighten_color(hoop_color, 0.2), width=2)
    
    def _lighten_color(self, hex_color, factor):
        """Lighten a hex color by factor (0-1)"""
        try:
            # Remove # if present
            hex_color = hex_color.lstrip('#')
            
            # Convert to RGB
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            
            # Lighten
            r = min(255, int(r + (255 - r) * factor))
            g = min(255, int(g + (255 - g) * factor))
            b = min(255, int(b + (255 - b) * factor))
            
            return f'#{r:02x}{g:02x}{b:02x}'
        except:
            return hex_color
    
    def _darken_color(self, hex_color, factor):
        """Darken a hex color by factor (0-1)"""
        try:
            # Remove # if present
            hex_color = hex_color.lstrip('#')
            
            # Convert to RGB
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            
            # Darken
            r = max(0, int(r * (1 - factor)))
            g = max(0, int(g * (1 - factor)))
            b = max(0, int(b * (1 - factor)))
            
            return f'#{r:02x}{g:02x}{b:02x}'
        except:
            return hex_color
