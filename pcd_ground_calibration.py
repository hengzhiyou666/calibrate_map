#!/usr/bin/env python3
"""
点云地面标定脚本：自动检测点云的地面平面，并矫正坐标系，使地面与 XY 平面平行。

依赖:
    pip install open3d numpy

用法示例（当前目录有 123.pcd 时）:
    # 自动生成 123_calibrated.pcd
    python3 pcd_ground_calibration.py 123.pcd

    # 手动指定输出文件名
    python3 pcd_ground_calibration.py 123.pcd --output my_calibrated.pcd

命名规则:
- 输入文件名为 <name>.pcd，且未显式指定 --output 时，
  输出文件名自动为: <name>_calibrated.pcd
"""

import argparse
from pathlib import Path

import numpy as np


def rotation_matrix_from_vectors(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """
    计算将 vec1 旋转到 vec2 的 3x3 旋转矩阵（罗德里格斯公式）。
    """
    v1 = vec1 / np.linalg.norm(vec1)
    v2 = vec2 / np.linalg.norm(vec2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    if np.isclose(dot, 1.0, atol=1e-6):
        return np.eye(3)
    if np.isclose(dot, -1.0, atol=1e-6):
        if not np.isclose(v1[0], 0.0, atol=1e-3):
            ortho = np.array([-(v1[1] + v1[2]) / v1[0], 1.0, 1.0])
        elif not np.isclose(v1[1], 0.0, atol=1e-3):
            ortho = np.array([1.0, -(v1[0] + v1[2]) / v1[1], 1.0])
        else:
            ortho = np.array([1.0, 1.0, -(v1[0] + v1[1]) / v1[2]])
        axis = ortho / np.linalg.norm(ortho)
        return -np.eye(3) + 2.0 * np.outer(axis, axis)

    axis = np.cross(v1, v2)
    sin_theta = np.linalg.norm(axis)
    axis = axis / sin_theta
    cos_theta = dot
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    R = np.eye(3) + K * sin_theta + (K @ K) * (1.0 - cos_theta)
    return R


def detect_ground_plane_and_align(
    input_pcd: Path,
    output_pcd: Path,
    voxel_size: float = 0.1,
    distance_threshold: float = 0.05,
    ransac_n: int = 3,
    num_iterations: int = 1000,
) -> None:
    try:
        import open3d as o3d
    except ImportError:
        raise SystemExit(
            "未找到 open3d，请先安装:\n    pip install open3d"
        )

    input_pcd = Path(input_pcd)
    if not input_pcd.exists():
        raise FileNotFoundError(f"点云文件不存在: {input_pcd}")

    print(f"[INFO] 读取点云: {input_pcd}")
    pcd = o3d.io.read_point_cloud(str(input_pcd))
    if len(pcd.points) == 0:
        raise ValueError("点云为空，无法处理。")

    print(f"[INFO] 原始点数: {len(pcd.points)}")

    # 可选下采样以加速 RANSAC
    if voxel_size and voxel_size > 0:
        print(f"[INFO] 体素下采样，voxel_size = {voxel_size}")
        pcd_sampled = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"[INFO] 下采样后点数: {len(pcd_sampled.points)}")
    else:
        pcd_sampled = pcd

    print(
        "[INFO] 使用 RANSAC 拟合地面平面，"
        f"distance_threshold={distance_threshold}, "
        f"ransac_n={ransac_n}, num_iterations={num_iterations}"
    )
    plane_model, inliers = pcd_sampled.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    a, b, c, d = plane_model
    print(f"[INFO] 拟合到的平面方程: {a:.4f} x + {b:.4f} y + {c:.4f} z + {d:.4f} = 0")
    print(f"[INFO] 地面内点数量: {len(inliers)}")

    # 平面法向量
    n = np.array([a, b, c], dtype=float)
    norm = np.linalg.norm(n)
    if norm < 1e-6:
        raise ValueError("平面法向量接近零，拟合失败。")
    n /= norm

    # 目标法向量设为 z 轴正方向
    target = np.array([0.0, 0.0, 1.0], dtype=float)
    R = rotation_matrix_from_vectors(n, target)

    # 说明：为了保证“坐标原点到地面的距离不变”，
    # 这里所有旋转均绕坐标原点 (0,0,0) 进行，不再绕点云质心。
    center = [0.0, 0.0, 0.0]
    print(f"[INFO] 以坐标原点 {center} 为中心进行旋转，以保持原点到地面的距离不变")

    print("[INFO] 应用第一次旋转，将地面法向对齐到 z 轴")
    pcd.rotate(R, center=center)

    # 旋转后做一次基于地面的“水平对齐”（绕 z 轴），
    # 让地面上的主方向尽量贴合 X 轴，减少整体“歪斜感”。
    pts = np.asarray(pcd.points)
    ground_band = 0.2
    ground_mask = np.abs(pts[:, 2] - pts[:, 2].min()) < ground_band
    if np.count_nonzero(ground_mask) > 100:
        xy = pts[ground_mask, :2]
        xy_mean = xy.mean(axis=0)
        xy_centered = xy - xy_mean
        cov = np.cov(xy_centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        main_dir = eigvecs[:, np.argmax(eigvals)]
        v_dir = np.array([main_dir[0], main_dir[1], 0.0], dtype=float)
        if np.linalg.norm(v_dir) > 1e-6:
            v_dir /= np.linalg.norm(v_dir)
            angle = np.arctan2(v_dir[1], v_dir[0])
            c = np.cos(-angle)
            s = np.sin(-angle)
            Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            print("[INFO] 进一步绕 z 轴旋转，使地面主方向贴合 X 轴（仍绕原点旋转）")
            pts = (Rz @ pts.T).T
        else:
            print("[WARN] 地面主方向长度过小，跳过水平对齐。")
    else:
        print("[WARN] 近地面点数量过少，跳过水平对齐。")

    # 注意：不再对 z 方向做整体平移，从而保持“原点到地面平面的距离”不变
    pcd.points = o3d.utility.Vector3dVector(pts)

    output_pcd = Path(output_pcd)
    print(f"[INFO] 保存矫正后的点云到: {output_pcd}")
    output_pcd.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(output_pcd), pcd):
        raise IOError(f"写入点云失败: {output_pcd}")

    print("[INFO] 处理完成。现在该点云的地面应与 XY 平面大致平行。")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="自动检测点云地面平面并矫正坐标系，使地面与 XY 平面平行。"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="123.pcd",
        help="输入点云（PCD/PLY 等，Open3D 支持的格式），默认 123.pcd",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="输出矫正后的点云文件；若不指定，将自动使用 calibrated_<输入文件名>.pcd",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.1,
        help="RANSAC 前的体素下采样分辨率 (m)，设为 0 关闭，默认 0.1",
    )
    parser.add_argument(
        "--distance_thresh",
        type=float,
        default=0.05,
        help="RANSAC 平面拟合的距离阈值 (m)，默认 0.05",
    )
    parser.add_argument(
        "--ransac_n",
        type=int,
        default=3,
        help="每次 RANSAC 采样的点数，默认 3",
    )
    parser.add_argument(
        "--num_iter",
        type=int,
        default=1000,
        help="RANSAC 迭代次数，默认 1000",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        stem = input_path.stem  # 例如 123.pcd -> 123
        # 输出名: 原名_calibrated.pcd
        output_path = input_path.with_name(f"{stem}_calibrated.pcd")

    detect_ground_plane_and_align(
        input_pcd=input_path,
        output_pcd=output_path,
        voxel_size=args.voxel_size,
        distance_threshold=args.distance_thresh,
        ransac_n=args.ransac_n,
        num_iterations=args.num_iter,
    )


if __name__ == "__main__":
    main()
