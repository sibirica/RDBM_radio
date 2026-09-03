# Generate random GMSH meshes for training

import os
import numpy as np
import gmsh
import random
from scipy import ndimage

def flatten_dim_tags(lst):
    """Recursively flattens nested sequences down into standard (dim, tag) integer tuples."""
    flat = []
    if isinstance(lst, (list, tuple)):
        if len(lst) == 2 and isinstance(lst[0], int) and isinstance(lst[1], int):
            flat.append((lst[0], lst[1]))
        else:
            for item in lst:
                flat.extend(flatten_dim_tags(item))
    return flat

class HighVarietyDatasetGenerator:
    def __init__(self, output_dir=None):
        self.output_dir = output_dir or os.path.join(os.getcwd(), "realistic_dataset")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _initialize_gmsh(self, model_name):
        """Standardized Gmsh initialization with exact tolerances."""
        if gmsh.isInitialized():
            gmsh.finalize()
        gmsh.initialize()
        gmsh.model.add(model_name)
        gmsh.option.setNumber("Geometry.Tolerance", 1e-7)
        gmsh.option.setNumber("Geometry.ToleranceBoolean", 1e-7)
        return gmsh.model.occ

    def _mesh_and_export(self, name, scale_factor, mesh_max=4.5):
        """Generates optimized 3D mesh and writes STL."""
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.15 * scale_factor) 
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_max * scale_factor)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 16)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

        try:
            gmsh.model.mesh.generate(3)
            out_path = os.path.join(self.output_dir, f"{name}.stl")
            gmsh.write(out_path)
            print(f"  Successfully mesh compiled: {out_path}")
        except Exception as e:
            print(f"  Mesh compilation warning: {e}")
        finally:
            if gmsh.isInitialized():
                gmsh.finalize()


    def populate_exact_abel_sector(self, occ, theta_start, theta_end, body_radius, body_thickness):
        sec_A_s = theta_start + np.radians(0.1)
        sec_A_e = theta_end - np.radians(0.1)
        center_theta = (sec_A_s + sec_A_e) / 2.0
        
        br_scale = body_radius / 80.0

        # Abel base sector cutout volume
        empty_section = []
        z_bottom = -2.5
        z_top = body_thickness + 15.0
        
        # Align cutout bounds (40mm to 80mm) to expose the full span of the flanking cones
        r_in = 40.0 * br_scale
        r_out = body_radius - 2.5
        
        x1_b, y1_b = r_in * np.cos(sec_A_s), r_in * np.sin(sec_A_s)
        x2_b, y2_b = r_out * np.cos(sec_A_s), r_out * np.sin(sec_A_s)
        x3_b, y3_b = r_out * np.cos(sec_A_e), r_out * np.sin(sec_A_e)
        x4_b, y4_b = r_in * np.cos(sec_A_e), r_in * np.sin(sec_A_e)

        p1_bottom = occ.addPoint(x1_b, y1_b, z_bottom)
        p2_bottom = occ.addPoint(x2_b, y2_b, z_bottom)
        p3_bottom = occ.addPoint(x3_b, y3_b, z_bottom)
        p4_bottom = occ.addPoint(x4_b, y4_b, z_bottom)
        p_center_bottom = occ.addPoint(0, 0, z_bottom)

        p1_top = occ.addPoint(x1_b, y1_b, z_top)
        p2_top = occ.addPoint(x2_b, y2_b, z_top)
        p3_top = occ.addPoint(x3_b, y3_b, z_top)
        p4_top = occ.addPoint(x4_b, y4_b, z_top)
        p_center_top = occ.addPoint(0, 0, z_top)

        l_side1_b = occ.addLine(p1_bottom, p2_bottom)
        l_side2_b = occ.addLine(p3_bottom, p4_bottom)
        arc_out_b = occ.addCircleArc(p2_bottom, p_center_bottom, p3_bottom)
        arc_in_b  = occ.addCircleArc(p4_bottom, p_center_bottom, p1_bottom)
        w_bottom  = occ.addWire([l_side1_b, arc_out_b, l_side2_b, arc_in_b])

        l_side1_t = occ.addLine(p1_top, p2_top)
        l_side2_t = occ.addLine(p3_top, p4_top)
        arc_out_t = occ.addCircleArc(p2_top, p_center_top, p3_top)
        arc_in_t  = occ.addCircleArc(p4_top, p_center_top, p1_top)
        w_top     = occ.addWire([l_side1_t, arc_out_t, l_side2_t, arc_in_t])

        arch_out = occ.addThruSections([w_bottom, w_top], makeSolid=True)
        arch_tag = arch_out[0][1]
        empty_section.append((3, int(arch_tag)))

        # Conical flanks (Calculated along rotated directions)
        cylinders = []
        zi = body_thickness / 2.0
        radius = 12.0 * br_scale
        length = 45.0 * br_scale
        
        # Calculate rotated center coordinates
        rc = 75.0 * br_scale
        dist_y = 35.0 * br_scale
        
        offsets = [
            (rc, dist_y),   # Left flank
            (rc, -dist_y),  # Right flank
            (82.0 * br_scale, 0.0) # Central flank
        ]
        
        for local_x, local_y in offsets:
            rot_x = local_x * np.cos(center_theta) - local_y * np.sin(center_theta)
            rot_y = local_x * np.sin(center_theta) + local_y * np.cos(center_theta)
            
            distance = np.hypot(rot_x, rot_y)
            ux, uy = -rot_x / distance, -rot_y / distance
            dx, dy = ux * length, uy * length
            
            cyl = occ.addCone(rot_x, rot_y, zi, dx, dy, 0, radius, 0.65 * radius)
            cylinders.append((3, cyl))

        # Concentric spheres (Rotated along centerline)
        spheres = []
        sphere_radii = np.array([4, 7, 10]) * br_scale
        curr_local_x = 45.0 * br_scale
        
        for r_sph in sphere_radii:
            local_x = curr_local_x + r_sph
            rot_x = local_x * np.cos(center_theta)
            rot_y = local_x * np.sin(center_theta)
            
            sph = occ.addSphere(rot_x, rot_y, zi, r_sph)
            spheres.append((3, sph))
            curr_local_x = local_x + r_sph - (2.0 * br_scale)

        # Flanking internal cylinder channels (Rotated along centerline)
        cylinder_radii = np.array([10, 7, 4]) * br_scale
        for flank_y in [dist_y, -dist_y]:
            start_x = 75.0 * br_scale
            start_y = flank_y
            local_dist = np.hypot(start_x, start_y)
            ux_local = -start_x / local_dist
            uy_local = -start_y / local_dist
            curr_local_x = start_x
            curr_local_y = start_y
            
            for r_cyl in cylinder_radii:
                h_len = r_cyl * 1.75
                rot_x = curr_local_x * np.cos(center_theta) - curr_local_y * np.sin(center_theta)
                rot_y = curr_local_x * np.sin(center_theta) + curr_local_y * np.cos(center_theta)
                dx = (ux_local * h_len) * np.cos(center_theta) - (uy_local * h_len) * np.sin(center_theta)
                dy = (ux_local * h_len) * np.sin(center_theta) + (uy_local * h_len) * np.cos(center_theta)
                cyl = occ.addCylinder(rot_x, rot_y, zi, dx, dy, 0, r_cyl)
                spheres.append((3, cyl))
                curr_local_x += ux_local * h_len
                curr_local_y += uy_local * h_len

        # Exact outer wall clearance cuts
        empty_section2 = []
        r_out_ext = body_radius + 10.0
        p1_b2 = occ.addPoint(body_radius * np.cos(sec_A_s), body_radius * np.sin(sec_A_s), z_bottom)
        p2_b2 = occ.addPoint(r_out_ext * np.cos(sec_A_s), r_out_ext * np.sin(sec_A_s), z_bottom)
        p3_b2 = occ.addPoint(r_out_ext * np.cos(sec_A_e), r_out_ext * np.sin(sec_A_e), z_bottom)
        p4_b2 = occ.addPoint(body_radius * np.cos(sec_A_e), body_radius * np.sin(sec_A_e), z_bottom)

        p1_t2 = occ.addPoint(body_radius * np.cos(sec_A_s), body_radius * np.sin(sec_A_s), z_top)
        p2_t2 = occ.addPoint(r_out_ext * np.cos(sec_A_s), r_out_ext * np.sin(sec_A_s), z_top)
        p3_t2 = occ.addPoint(r_out_ext * np.cos(sec_A_e), r_out_ext * np.sin(sec_A_e), z_top)
        p4_t2 = occ.addPoint(body_radius * np.cos(sec_A_e), body_radius * np.sin(sec_A_e), z_top)

        l1_b2 = occ.addLine(p1_b2, p2_b2)
        l2_b2 = occ.addLine(p3_b2, p4_b2)
        arc_out_b2 = occ.addCircleArc(p2_b2, p_center_bottom, p3_b2)
        arc_in_b2  = occ.addCircleArc(p4_b2, p_center_bottom, p1_b2)
        w_bottom2  = occ.addWire([l1_b2, arc_out_b2, l2_b2, arc_in_b2])

        l1_t2 = occ.addLine(p1_t2, p2_t2)
        l2_t2 = occ.addLine(p3_t2, p4_t2)
        arc_out_t2 = occ.addCircleArc(p2_t2, p_center_top, p3_t2)
        arc_in_t2  = occ.addCircleArc(p4_t2, p_center_top, p1_t2)
        w_top2     = occ.addWire([l1_t2, arc_out_t2, l2_t2, arc_in_t2])

        arch_out2 = occ.addThruSections([w_bottom2, w_top2], makeSolid=True)
        empty_section2.append((3, arch_out2[0][1]))

        return {
            "outer_cutouts": empty_section,
            "additive_cylinders": cylinders,
            "inner_subtractions": spheres,
            "top_clearance_cuts": empty_section2
        }

    def _ray_direction_to_source(self, x, y, z_source):
        dx, dy, dz = -x, -y, z_source
        mag = np.sqrt(dx**2 + dy**2 + dz**2)
        return (dx/mag, dy/mag, dz/mag) if mag > 0 else (0.0, 0.0, 1.0)

    def _add_hourglass_box(self, occ, x, y, z, dx, dy, dz):
        waist_factor = 0.98
        z_bottom, z_mid, z_top = z, z + (dz / 2.0), z + dz
        dx_mid, dy_mid = dx * waist_factor, dy * waist_factor

        p1 = occ.addPoint(x - dx/2.0, y - dy/2.0, z_bottom)
        p2 = occ.addPoint(x + dx/2.0, y - dy/2.0, z_bottom)
        p3 = occ.addPoint(x + dx/2.0, y + dy/2.0, z_bottom)
        p4 = occ.addPoint(x - dx/2.0, y + dy/2.0, z_bottom)
        l1, l2, l3, l4 = occ.addLine(p1, p2), occ.addLine(p2, p3), occ.addLine(p3, p4), occ.addLine(p4, p1)
        w_bottom = occ.addWire([l1, l2, l3, l4])
        
        pm1 = occ.addPoint(x - dx_mid/2.0, y - dy_mid/2.0, z_mid)
        pm2 = occ.addPoint(x + dx_mid/2.0, y - dy_mid/2.0, z_mid)
        pm3 = occ.addPoint(x + dx_mid/2.0, y + dy_mid/2.0, z_mid)
        pm4 = occ.addPoint(x - dx_mid/2.0, y + dy_mid/2.0, z_mid)
        lm1, lm2, lm3, lm4 = occ.addLine(pm1, pm2), occ.addLine(pm2, pm3), occ.addLine(pm3, pm4), occ.addLine(pm4, pm1)
        w_mid = occ.addWire([lm1, lm2, lm3, lm4])

        pt1 = occ.addPoint(x - dx/2.0, y - dy/2.0, z_top)
        pt2 = occ.addPoint(x + dx/2.0, y - dy/2.0, z_top)
        pt3 = occ.addPoint(x + dx/2.0, y + dy/2.0, z_top)
        pt4 = occ.addPoint(x - dx/2.0, y + dy/2.0, z_top)
        lt1, lt2, lt3, lt4 = occ.addLine(pt1, pt2), occ.addLine(pt2, pt3), occ.addLine(pt3, pt4), occ.addLine(pt4, pt1)
        w_top = occ.addWire([lt1, lt2, lt3, lt4])

        hourglass_tags = occ.addThruSections([w_bottom, w_mid, w_top], -1, True, False)
        volume_tag = hourglass_tags[0][1]
        return volume_tag

    def _create_angled_rect_cut(self, occ, x, y, dx, dy, body_thickness, z_source, overrun=2.5):
        """Standardized cutter with explicit double overruns (2.5mm)."""
        cx, cy = x + dx / 2.0, y + dy / 2.0
        rdx, rdy, rdz = self._ray_direction_to_source(cx, cy, z_source)
        cos_theta = abs(rdz)
        path_len = body_thickness / cos_theta if cos_theta > 0.0 else body_thickness
        drill_len = path_len + (2.0 * overrun)
        
        box_tag = self._add_hourglass_box(occ, 0.0, 0.0, 0.0, dx, dy, drill_len)
        angle_y = np.atan2(rdx, rdz)
        angle_x = -np.atan2(rdy, np.hypot(rdx, rdz))
        
        occ.rotate([(3, box_tag)], 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, angle_x)
        occ.rotate([(3, box_tag)], 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, angle_y)
        
        x_start = cx - rdx * overrun
        y_start = cy - rdy * overrun
        z_start = -rdz * overrun
        occ.translate([(3, box_tag)], x_start, y_start, z_start)
        return box_tag

    def _drill_cylinder_toward_source(self, occ, x, y, radius, body_thickness, z_source, depth_frac=1.0, overrun=2.5):
        """Standardized cylinder cutter with explicit double overruns."""
        rdx, rdy, rdz = self._ray_direction_to_source(x, y, z_source)
        cos_theta = np.abs(rdz)
        path_len = body_thickness / cos_theta if cos_theta > 0 else body_thickness
        drill_len = (path_len * depth_frac) + (2 * overrun)
        x_start = x - rdx * overrun
        y_start = y - rdy * overrun
        z_start = -rdz * overrun 
        
        cyl = occ.addCylinder(x_start, y_start, z_start, rdx*drill_len, rdy*drill_len, rdz*drill_len, radius)
        return (3, cyl)

    def _create_sector_step(self, occ, r_in, r_out, theta_start, theta_end, body_thickness, height):
        p_center = occ.addPoint(0, 0, body_thickness)
        p1 = occ.addPoint(r_in * np.cos(theta_start), r_in * np.sin(theta_start), body_thickness)
        p2 = occ.addPoint(r_out * np.cos(theta_start), r_out * np.sin(theta_start), body_thickness)
        p3 = occ.addPoint(r_out * np.cos(theta_end), r_out * np.sin(theta_end), body_thickness)
        p4 = occ.addPoint(r_in * np.cos(theta_end), r_in * np.sin(theta_end), body_thickness)
        
        l1 = occ.addLine(p1, p2)
        arc_out = occ.addCircleArc(p2, p_center, p3)
        l2 = occ.addLine(p3, p4)
        arc_in = occ.addCircleArc(p4, p_center, p1)
        
        w_tag = occ.addWire([l1, arc_out, l2, arc_in])
        surface_tag = occ.addPlaneSurface([w_tag])
        extruded_entities = occ.extrude([(2, surface_tag)], 0, 0, -float(height))
        
        volume_tag = None
        for dim, tag in extruded_entities:
            if dim == 3:
                volume_tag = tag
                break
                
        occ.remove([
            (0, p_center), (0, p1), (0, p2), (0, p3), (0, p4),
            (1, l1), (1, arc_out), (1, l2), (1, arc_in), (1, w_tag), (2, surface_tag)
        ], recursive=False)
        return (3, volume_tag)

    def populate_concentric_stepwedge(self, occ, r_in, r_out, body_thickness, theta_start, theta_end, num_steps=6):
        steps = []
        angles = np.linspace(theta_start + 0.01, theta_end - 0.01, num_steps + 1)
        max_safe_depth = body_thickness * 0.90  
        heights = np.linspace(3.0 * (body_thickness / 25.508), max_safe_depth, num_steps)
        for i in range(num_steps):
            step = self._create_sector_step(occ, r_in, r_out, angles[i] + 0.005, angles[i+1] - 0.005, body_thickness, heights[i])
            steps.append(step)
        return steps

    def populate_polar_pinholes(self, occ, r_in, r_out, theta_start, theta_end, body_thickness, z_source, num_holes=8):
        holes = []
        buffer = np.radians(3.5) 
        angles = np.linspace(theta_start + buffer, theta_end - buffer, num_holes)
        radii_levels = [r_in + (r_out - r_in) * 0.35, r_in + (r_out - r_in) * 0.70]
        
        for r_band in radii_levels:
            radii = np.linspace(3.5, 0.8, num_holes) * (body_thickness / 25.508)
            for i, ang in enumerate(angles):
                if i > 0:
                    delta_theta = angles[i] - angles[i-1]
                    pitch = r_band * delta_theta
                    safety_factor = 1.05 if r_band < (r_in + (r_out - r_in) * 0.5) else 1.2
                    
                    if pitch < (radii[i] + radii[i-1]) * safety_factor:
                        continue 
                
                px = r_band * np.cos(ang)
                py = r_band * np.sin(ang)
                pin = self._drill_cylinder_toward_source(occ, px, py, radii[i], body_thickness, z_source)
                holes.append(pin)
        return holes

    def populate_siemens_star(self, occ, r_in, r_out, body_thickness):
        """Generates a high-precision, crash-proof radial Siemens Star wheel cut."""
        star_cuts = []
        num_spokes = 16
        delta_theta = (2.0 * np.pi) / (num_spokes * 2)
        
        for i in range(num_spokes):
            theta_s = i * 2 * delta_theta
            theta_e = theta_s + delta_theta
            
            # Straight parallel extrusion cut (with overrun)
            overrun = 2.5
            spoke_wedge = self._create_sector_step(occ, r_in, r_out, theta_s, theta_e, body_thickness, body_thickness + (2 * overrun))
            # Offset downward to ensure it cuts all the way through
            occ.translate([spoke_wedge], 0, 0, -overrun)
            star_cuts.append(spoke_wedge)
            
        return star_cuts

    def populate_central_mura(self, occ, cx, cy, side_len, body_thickness):
        """
        Generates a non-intersecting parallel vertical MURA coded mask.
        This avoids the Boolean collapses caused by converging 3D radial loft paths.
        """
        mura_cuts = []
        L = 29
        A = self._generate_mura_mask(L)
        
        pixel_size = side_len / L
        shift_x, shift_y = cx - (side_len / 2.0), cy - (side_len / 2.0)
        
        labeled, num_feat = ndimage.label(A == 0, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
        bboxes = ndimage.find_objects(labeled)
        
        for i, box in enumerate(bboxes):
            if box is None: continue
            r_slice, c_slice = box[0], box[1]
            bx = c_slice.start * pixel_size + shift_x
            by = r_slice.start * pixel_size + shift_y
            bw = (c_slice.stop - c_slice.start) * pixel_size
            bh = (r_slice.stop - r_slice.start) * pixel_size
            
            # Straight parallel extrusion cut through the plate along the Z-axis (with overrun)
            overrun = 2.5
            box_tag = occ.addBox(bx, by, -overrun, bw, bh, body_thickness + (2 * overrun))
            mura_cuts.append((3, box_tag))
        return mura_cuts

    def build_linear_rows_layout(self, occ, body_radius, body_thickness, z_source, variety_type="A"):
        """Constructs highly distinct parallel row calibration designs with safe feature spacing."""
        cuts = []
        max_safe_depth = body_thickness * 0.90
        
        br_scale = body_radius / 80.0
        bt_scale = body_thickness / 25.508
        
        if variety_type == "A":
            # Outer Ring Band (Linepair slots and Stepwedges safely shifted)
            step_h1 = np.clip(np.array([4, 15, 19, 21, 23, 21, 16, 14, 8, 3]) * bt_scale, 0.5, max_safe_depth)
            box_w = 8.0 * br_scale
            box_l = 12.0 * br_scale
            
            xi = -40.0 * br_scale
            for h in step_h1:
                # Placed safely in the upper hemisphere
                step = occ.addBox(xi, 55.0 * br_scale, body_thickness, box_w, box_l, -h)
                cuts.append((3, step))
                step = occ.addBox(xi, -20.0 * br_scale, body_thickness, box_w, box_l, -h)
                cuts.append((3, step))
                xi += box_w
                
            square_size = 16.0 * br_scale
            resolutions = np.array([10.0, 8.0, 4.0, 2.0, 1.0, 0.5]) * br_scale
            xi = -60.0 * br_scale
            for r_sz in resolutions:
                num_slots = int(square_size / r_sz)
                for s in range(num_slots):
                    if s % 2 == 0:
                        rx = xi + (s * r_sz)
                        # Offset slots safely below the stepwedges
                        tool = self._create_angled_rect_cut(occ, rx, 25.0 * br_scale, r_sz, square_size, body_thickness, z_source)
                        cuts.append((3, tool))
                xi += square_size + (4.0 * br_scale)
                
        elif variety_type == "B":
            # Asymmetrical Skewed Bands
            step_h2 = np.clip(np.array([2, 8, 17, 21, 24, 18, 12, 6, 2]) * bt_scale, 0.5, max_safe_depth)
            box_w = 9.0 * br_scale
            box_l = 10.0 * br_scale
            
            xi = -45.0 * br_scale
            for h in step_h2:
                step = occ.addBox(xi, 40.0 * br_scale, body_thickness, box_w, box_l, -h)
                cuts.append((3, step))
                xi += box_w
                
        else:
            # Concentric/Linear Hybrid Array
            step_h3 = np.clip(np.array([3, 10, 18, 22, 19, 12, 5]) * bt_scale, 0.5, max_safe_depth)
            box_w = 11.0 * br_scale
            box_l = 14.0 * br_scale
            
            xi = -38.0 * br_scale
            for h in step_h3:
                step = occ.addBox(xi, -55.0 * br_scale, body_thickness, box_w, box_l, -h)
                cuts.append((3, step))
                xi += box_w

        # Decreasing rows of pinholes safely aligned in outer concentric band
        pin_radii = np.array([3.0, 2.75, 2.5, 2.25, 2.0, 1.75, 1.5, 1.25]) * br_scale
        pin_y_levels = np.array([65.0, 55.0, 45.0]) * br_scale 
        
        for y_lvl in pin_y_levels:
            xi_pin = 25.0 * br_scale
            for r_pin in pin_radii:
                pin = self._drill_cylinder_toward_source(occ, xi_pin, -y_lvl, r_pin, body_thickness, z_source)
                cuts.append(pin)
                xi_pin -= (r_pin + 5.0 * br_scale)

        return cuts

    def _generate_mura_mask(self, L):
        A = np.zeros((L, L), dtype=int)
        q_res = np.zeros(L, dtype=int)
        for x in range(1, L):
            q_res[(x * x) % L] = 1
        for i in range(L):
            for j in range(L):
                if i == 0 and j > 0: A[i, j] = 0
                elif j == 0: A[i, j] = 1
                else:
                    A[i, j] = 1 if (q_res[i] + q_res[j]) % 2 == 0 else 0
        return A

    def generate_static_phantom_config(self, index, scale_factor=1.0):
        name = f"packed_calibration_target_{index}"
        occ = self._initialize_gmsh(name)
        body_radius = 80.0 * scale_factor
        body_thickness = (25.0 + 0.508) * scale_factor
        sod = 550.0 * scale_factor
        z_source = sod + body_thickness
        central_bore_r = 5.0 * scale_factor
        print(f"Generating Pre-configured Target Config #{index}...")
        body_cyl = occ.addCylinder(0, 0, 0, 0, 0, body_thickness, body_radius)
        body_result = [(3, body_cyl)]
        occ.synchronize()

        subtractions = []
        additions = []

        if index in [1, 5, 6, 8]:
            bore = occ.addCylinder(0, 0, -1.0, 0, 0, body_thickness + 2.0, central_bore_r)
            subtractions.append((3, bore))
        elif index in [3, 10]:
            bore = occ.addCylinder(0, 0, -1.0, 0, 0, body_thickness + 2.0, central_bore_r)
            subtractions.append((3, bore))
            star_elements = self.populate_siemens_star(occ, central_bore_r + 1.0 * scale_factor, 35.0 * scale_factor, body_thickness + 5)
            subtractions.extend(star_elements)
        else:
            mura_size = 45.0 * scale_factor
            mura_elements = self.populate_central_mura(occ, 0.0, 0.0, mura_size + 10, body_thickness)
            subtractions.extend(mura_elements)

        r_in = 46.0 * scale_factor  
        r_out = body_radius - 2.5   
        sectors = []
        abel_sector_params = None

        if index == 1:
            sectors = [
                (np.pi/2.0 + 0.01, np.pi - 0.01, "stepwedge"),
                (np.pi + 0.01, 1.5*np.pi - 0.01, "pinholes"),
                (1.5*np.pi + 0.01, 2.0*np.pi - 0.01, "stepwedge")
            ]
            abel_sector_params = (0.01, np.pi/2.0 - 0.01)
        elif index == 2:
            sectors = [
                (np.pi/4.0 + 0.01, np.pi/4.0 + np.pi/2.0 - 0.01, "pinholes"),
                (np.pi/4.0 + np.pi + 0.01, np.pi/4.0 + 1.5*np.pi - 0.01, "stepwedge"),
                (np.pi/4.0 + 1.5*np.pi + 0.01, np.pi/4.0 + 2.0*np.pi - 0.01, "pinholes")
            ]
            abel_sector_params = (np.pi/4.0 + np.pi/2.0 + 0.01, np.pi/4.0 + np.pi - 0.01)
        elif index == 3:
            sectors = [
                (2.0*np.pi/3.0 + 0.01, 4.0*np.pi/3.0 - 0.01, "stepwedge"),
                (4.0*np.pi/3.0 + 0.01, 2.0*np.pi - 0.01, "pinholes")
            ]
            abel_sector_params = (0.01, 2.0*np.pi/3.0 - 0.01)
        elif index == 4:
            sectors = [
                (np.pi/6.0 + 0.01, np.pi/6.0 + 2.0*np.pi/3.0 - 0.01, "stepwedge"),
                (np.pi/6.0 + 2.0*np.pi/3.0 + 0.01, np.pi/6.0 + 4.0*np.pi/3.0 - 0.01, "pinholes")
            ]
            abel_sector_params = (np.pi/6.0 + 4.0*np.pi/3.0 + 0.01, np.pi/6.0 + 2.0*np.pi - 0.01)
        elif index == 5:
            sectors = [
                (np.pi + 0.01, 2.0*np.pi - 0.01, "stepwedge")
            ]
            abel_sector_params = (0.01, np.pi - 0.01)
        elif index == 6:
            linear_elements = self.build_linear_rows_layout(occ, body_radius, body_thickness, z_source, variety_type="A")
            subtractions.extend(linear_elements)
        elif index == 7:
            linear_elements = self.build_linear_rows_layout(occ, body_radius, body_thickness, z_source, variety_type="B")
            subtractions.extend(linear_elements)
        elif index == 8:
            sectors = [
                (np.pi/2.0 + 0.01, np.pi - 0.01, "pinholes"),
                (1.5*np.pi + 0.01, 2.0*np.pi - 0.01, "stepwedge")
            ]
            abel_sector_params = [(0.01, np.pi/2.0 - 0.01), (np.pi + 0.01, 1.5*np.pi - 0.01)]
        elif index == 9:
            abel_sector_params = [
                (np.pi/3.0 - np.radians(10) + 0.01, 2.0*np.pi/3.0 + np.radians(10) - 0.01),  
                (4.0*np.pi/3.0 - np.radians(10) + 0.01, 5.0*np.pi/3.0 + np.radians(10) - 0.01)
            ]
            sectors = [
                (5.0*np.pi/3.0 + np.radians(10) + 0.01, 2.0*np.pi - 0.01, "pinholes"), 
                (0.0 + 0.01, np.pi/3.0 - np.radians(10) - 0.01, "pinholes"),      
                (2.0*np.pi/3.0 + np.radians(10) + 0.01, np.pi - 0.01, "stepwedge"),  
                (np.pi + 0.01, 4.0*np.pi/3.0 - np.radians(10) - 0.01, "stepwedge")  
            ]
        else:
            sectors = [
                (np.pi/4.0 + 0.01, np.pi/4.0 + np.pi/2.0 - 0.01, "stepwedge"),
                (np.pi/4.0 + np.pi + 0.01, np.pi/4.0 + 1.5*np.pi - 0.01, "stepwedge"),
                (np.pi/4.0 + 1.5*np.pi + 0.01, np.pi/4.0 + 2.0*np.pi - 0.01, "stepwedge"),
                (np.pi/4.0 + np.pi/2.0 + 0.01, np.pi/4.0 + np.pi - 0.01, "stepwedge")
            ]

        # Allocate sector elements
        for th_s, th_e, sect_type in sectors:
            if sect_type == "stepwedge":
                subtractions.extend(self.populate_concentric_stepwedge(occ, r_in, r_out, body_thickness, th_s, th_e))
            elif sect_type == "pinholes":
                subtractions.extend(self.populate_polar_pinholes(occ, r_in, r_out, th_s, th_e, body_thickness, z_source))

        flat_subs = flatten_dim_tags(subtractions)
        flat_adds = flatten_dim_tags(additions)
        occ.synchronize()

        if abel_sector_params is not None:
            # Handle list vs single tuple configs for abel parameter variables
            if not isinstance(abel_sector_params, list):
                abel_sector_params = [abel_sector_params]
                
            for th_s, th_e in abel_sector_params:
                abel_elements = self.populate_exact_abel_sector(occ, th_s, th_e, body_radius, body_thickness)
                body_result, _ = occ.cut(body_result, flatten_dim_tags(abel_elements["outer_cutouts"]))
                occ.synchronize()
                body_result, _ = occ.fuse(body_result, flatten_dim_tags(abel_elements["additive_cylinders"]))
                occ.synchronize()
                body_result, _ = occ.cut(body_result, flatten_dim_tags(abel_elements["inner_subtractions"]))
                occ.synchronize()
                body_result, _ = occ.cut(body_result, flatten_dim_tags(abel_elements["top_clearance_cuts"]))
                occ.synchronize()

        # Cut remaining general subtractive elements (MURA Core, Bores, wedges, linepairs, rows)
        if flat_subs:
            body_result, _ = occ.cut(body_result, flat_subs)
        occ.synchronize()

        self._mesh_and_export(name, scale_factor)

if __name__ == "__main__":
    try:
        n_objects = int(input("Enter number of realistic phantoms to generate (1 to 10): "))
    except ValueError:
        n_objects = 3
        print("Invalid input. Defaulting to 3 objects.")

    n_objects = min(max(n_objects, 1), 10)
    generator = HighVarietyDatasetGenerator()
    print(f"\nInitiating high-variety target generation for {n_objects} objects...")
    
    for i in range(1, n_objects + 1):
        scale = round(random.uniform(0.6, 1.4), 2)
        generator.generate_static_phantom_config(i, scale_factor=scale)

    print("\n--- Calibration Target Generation Successfully Completed. ---")