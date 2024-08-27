import xml.etree.ElementTree as ET
import xml.dom.minidom
import math
import numpy as np

input = "iris_environments/assets/models/iiwa_description/iiwa7/iiwa7_with_box_collision.sdf"
output = "iris_environments/assets/models/iiwa_description/iiwa7/iiwa7_with_box_collision.urdf"

def convert_pose(pose_string):
    pose = [float(x) for x in pose_string.split()]
    return pose[:3], pose[3:]  # Return [x, y, z], [roll, pitch, yaw]

def add_vectors(v1, v2):
    return [a + b for a, b in zip(v1, v2)]

def subtract_vectors(v1, v2):
    return [a - b for a, b in zip(v1, v2)]

def map_joint_type(sdf_type):
    type_map = {
        "revolute": "continuous",
        "prismatic": "prismatic",
        "fixed": "fixed",
        "continuous": "continuous",
        "ball": "floating",
        "universal": "floating"
    }
    return type_map.get(sdf_type, "fixed")

def pose_to_matrix(pose):
    x, y, z, roll, pitch, yaw = pose
    c = np.cos
    s = np.sin
    R = np.array([
        [c(yaw)*c(pitch), -s(yaw)*c(roll)+c(yaw)*s(pitch)*s(roll), s(yaw)*s(roll)+c(yaw)*s(pitch)*c(roll)],
        [s(yaw)*c(pitch), c(yaw)*c(roll)+s(yaw)*s(pitch)*s(roll), -c(yaw)*s(roll)+s(yaw)*s(pitch)*c(roll)],
        [-s(pitch), c(pitch)*s(roll), c(pitch)*c(roll)]
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T

def matrix_to_pose(matrix):
    x, y, z = matrix[:3, 3]
    R = matrix[:3, :3]
    pitch = math.atan2(-R[2, 0], math.sqrt(R[2, 1]**2 + R[2, 2]**2))
    if abs(pitch) == math.pi/2:
        yaw = 0
        roll = math.atan2(R[0, 1], R[1, 1])
    else:
        yaw = math.atan2(R[1, 0], R[0, 0])
        roll = math.atan2(R[2, 1], R[2, 2])
    return [x, y, z, roll, pitch, yaw]

def parse_pose(pose_string):
    return np.array([float(x) for x in pose_string.split()])

def split_pose(pose_array):
    return list(pose_array[:3]), list(pose_array[3:])

def convert_sdf_to_urdf(sdf_file, urdf_file):
    # Parse the SDF file
    tree = ET.parse(sdf_file)
    root = tree.getroot()

    # Create the root of the URDF file
    urdf_root = ET.Element("robot", name=root.find("model").attrib["name"])

    # First pass: collect all link poses
    link_poses = {}
    for link in root.findall(".//link"):
        link_name = link.attrib["name"]
        link_pose = link.find("pose")
        if link_pose is not None:
            link_poses[link_name] = parse_pose(link_pose.text)
        else:
            link_poses[link_name] = np.zeros(6)

    # Second pass: create URDF links
    for link in root.findall(".//link"):
        urdf_link = ET.SubElement(urdf_root, "link", name=link.attrib["name"])

        xyz, rpy = split_pose(link_poses[link.attrib["name"]])
        ET.SubElement(urdf_link, "origin", xyz=f"{xyz[0]} {xyz[1]} {xyz[2]}", rpy=f"{rpy[0]} {rpy[1]} {rpy[2]}")

        # Convert inertial
        inertial = link.find("inertial")
        if inertial is not None:
            urdf_inertial = ET.SubElement(urdf_link, "inertial")
            mass = inertial.find("mass").text
            ET.SubElement(urdf_inertial, "mass", value=mass)

            inertia = inertial.find("inertia")
            urdf_inertia = ET.SubElement(urdf_inertial, "inertia", 
                                         ixx=inertia.find("ixx").text,
                                         ixy=inertia.find("ixy").text,
                                         ixz=inertia.find("ixz").text,
                                         iyy=inertia.find("iyy").text,
                                         iyz=inertia.find("iyz").text,
                                         izz=inertia.find("izz").text)

        # Convert visual elements
        for visual in link.findall("visual"):
            urdf_visual = ET.SubElement(urdf_link, "visual")
            if visual.find("pose") is not None:
                v_xyz, v_rpy = convert_pose(visual.find("pose").text)
                ET.SubElement(urdf_visual, "origin", xyz=f"{v_xyz[0]} {v_xyz[1]} {v_xyz[2]}", rpy=f"{v_rpy[0]} {v_rpy[1]} {v_rpy[2]}")

            geometry = visual.find("geometry")
            urdf_geometry = ET.SubElement(urdf_visual, "geometry")
            
            if geometry.find("mesh") is not None:
                mesh = geometry.find("mesh")
                ET.SubElement(urdf_geometry, "mesh", filename=mesh.find("uri").text)
            elif geometry.find("box") is not None:
                box = geometry.find("box")
                ET.SubElement(urdf_geometry, "box", size=box.find("size").text)

            # Add material information
            material = visual.find("material")
            if material is not None:
                urdf_material = ET.SubElement(urdf_visual, "material", name=f"{link.attrib['name']}_material")
                diffuse = material.find("diffuse")
                if diffuse is not None:
                    color_values = diffuse.text.split()
                    if len(color_values) == 4:  # RGBA
                        color_str = " ".join(color_values)
                        ET.SubElement(urdf_material, "color", rgba=color_str)

        # Convert collision elements
        for collision in link.findall("collision"):
            urdf_collision = ET.SubElement(urdf_link, "collision", name=collision.attrib["name"])
            if collision.find("pose") is not None:
                xyz, rpy = convert_pose(collision.find("pose").text)
                ET.SubElement(urdf_collision, "origin", xyz=f"{xyz[0]} {xyz[1]} {xyz[2]}", rpy=f"{rpy[0]} {rpy[1]} {rpy[2]}")

            geometry = collision.find("geometry")
            box = geometry.find("box")
            if box is not None:
                size = box.find("size").text
                ET.SubElement(urdf_collision, "geometry").append(
                    ET.Element("box", size=size)
                )

     # Third pass: create URDF joints
    for sdf_joint in root.findall(".//joint"):
        urdf_joint = ET.SubElement(urdf_root, "joint")
        urdf_joint.set("name", sdf_joint.get("name"))
        urdf_joint.set("type", sdf_joint.get("type"))

        parent_link = sdf_joint.find("parent").text
        child_link = sdf_joint.find("child").text

        parent = ET.SubElement(urdf_joint, "parent")
        parent.set("link", parent_link)
        child = ET.SubElement(urdf_joint, "child")
        child.set("link", child_link)

        parent_pose = pose_to_matrix(link_poses[parent_link])
        child_pose = pose_to_matrix(link_poses[child_link])
        joint_pose = np.linalg.inv(parent_pose) @ child_pose
        
        joint_xyz, joint_rpy = split_pose(matrix_to_pose(joint_pose))

        ET.SubElement(urdf_joint, "origin", 
                      xyz=f"{joint_xyz[0]} {joint_xyz[1]} {joint_xyz[2]}", 
                      rpy=f"{joint_rpy[0]} {joint_rpy[1]} {joint_rpy[2]}")

        sdf_axis = sdf_joint.find("axis/xyz")
        if sdf_axis is not None:
            axis = ET.SubElement(urdf_joint, "axis")
            axis.set("xyz", sdf_axis.text)

    # Convert the ElementTree to a string and prettify
    xml_string = ET.tostring(urdf_root, encoding="unicode")
    dom = xml.dom.minidom.parseString(xml_string)
    pretty_xml = dom.toprettyxml(indent="  ")

    # Write the prettified XML to the output file
    with open(urdf_file, "w") as f:
        f.write(pretty_xml)

# Usage
convert_sdf_to_urdf(input, output)