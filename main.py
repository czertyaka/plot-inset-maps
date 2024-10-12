#!/bin/env python3

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.patheffects as path_effects
import osmnx
import shapely
import geopandas
import re


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Plot points on maps and embedded inset_maps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        """,
    )
    parser.add_argument(
        "-i", "--input", required=True, type=str, help="path to input JSON file"
    )
    parser.add_argument(
        "-o", "--output", required=False, type=str, help="path to output PNG file"
    )
    return parser.parse_args()


def read_input(filePath):
    with open(filePath, "r") as file:
        return json.load(file)


def validate_input_structure(input):
    assert "main" in input.keys(), f"input={input}"
    maps = [input["main"]]
    if "inset_maps" in input.keys():
        for inset_map in input["inset_maps"]:
            for key in inset_map.keys():
                assert key in ("layout", "map"), f"inset_map={inset_map}"
            assert "layout" in inset_map.keys(), f"inset_map={inset_map}"
            assert "map" in inset_map.keys(), f"inset_map={inset_map}"
            layout = inset_map["layout"]
            for key in layout.keys():
                assert key in ("x", "y", "scale")
            assert "x" in inset_map["layout"].keys()
            assert "y" in inset_map["layout"].keys()
            assert "scale" in inset_map["layout"].keys()
            maps.append(inset_map["map"])
    points = []
    labels = []
    for map in maps:
        for key in map.keys():
            assert key in ("bbox", "points", "labels")
        assert "bbox" in map.keys(), f"map={map}"
        bbox = map["bbox"]
        for key in bbox.keys():
            assert key in ("north", "south", "east", "west"), f"bbox={bbox}"
        assert "north" in bbox.keys(), f"bbox={bbox}"
        assert "south" in bbox.keys(), f"bbox={bbox}"
        assert "east" in bbox.keys(), f"bbox={bbox}"
        assert "west" in bbox.keys(), f"bbox={bbox}"
        if "points" in map.keys():
            [points.append(point) for point in map["points"]]
        if "labels" in map.keys():
            [labels.append(label) for label in map["labels"]]
    for point in points:
        for key in point.keys():
            assert key in ("lon", "lat", "name"), f"point={point}"
        assert "lon" in point.keys(), f"point={point}"
        assert "lat" in point.keys(), f"point={point}"
        assert "name" in point.keys(), f"point={point}"
    for label in labels:
        for key in label.keys():
            assert key in ("lon", "lat", "text", "color"), f"label={label}"
        assert "lon" in label.keys(), f"label={label}"
        assert "lat" in label.keys(), f"label={label}"
        assert "text" in label.keys(), f"label={label}"


def add_input_default_values(input):
    if "inset_maps" not in input.keys():
        input["inset_maps"] = []
    maps = [input["main"]]
    for map in input["inset_maps"]:
        maps.append(map)
    for map in maps:
        if "points" not in map.keys():
            map["points"] = []
        if "labels" not in map.keys():
            map["labels"] = []


def convert_coordinate(coordinate):
    pattern = re.compile("^[0-9]{1,3}°[0-9]{2}'[0-9]{2}(\\.[0-9]+)?\"[NSEW]$")
    if pattern.match(coordinate):
        splitted = re.split("[°'\"]", coordinate)
        degree = float(splitted[0])
        minutes = float(splitted[1])
        seconds = float(splitted[2])
        sign = 1 if splitted[3] == "N" or splitted[3] == "E" else -1
        return sign * (degree + minutes / 60 + seconds / 3600)
    else:
        return float(coordinate)


def convert_input_values(input):
    maps = [input["main"]]
    layouts = []
    points = []
    labels = []
    for inset_map in input["inset_maps"]:
        maps.append(inset_map["map"])
        layouts.append(inset_map["layout"])
    for map in maps:
        bbox = map["bbox"]
        bbox["north"] = convert_coordinate(bbox["north"])
        bbox["south"] = convert_coordinate(bbox["south"])
        bbox["east"] = convert_coordinate(bbox["east"])
        bbox["west"] = convert_coordinate(bbox["west"])
        points += map["points"]
        labels += map["labels"]
    for point in points:
        point["lat"] = convert_coordinate(point["lat"])
        point["lon"] = convert_coordinate(point["lon"])
    for label in labels:
        label["lat"] = convert_coordinate(label["lat"])
        label["lon"] = convert_coordinate(label["lon"])
    for layout in layouts:
        layout["x"] = float(layout["x"])
        layout["y"] = float(layout["y"])
        layout["scale"] = float(layout["scale"])


def validate_coordinate(coordinate):
    assert coordinate >= -180 and coordinate <= 180, f"coordinate={coordinate}"


def validate_input_values(input):
    points = []
    validate_coordinate(input["main"]["bbox"]["west"])
    validate_coordinate(input["main"]["bbox"]["south"])
    validate_coordinate(input["main"]["bbox"]["east"])
    validate_coordinate(input["main"]["bbox"]["north"])
    main_box = shapely.box(
        input["main"]["bbox"]["west"],
        input["main"]["bbox"]["south"],
        input["main"]["bbox"]["east"],
        input["main"]["bbox"]["north"],
    )
    for point in input["main"]["points"]:
        points.append((main_box, point))
    for label in input["main"]["labels"]:
        points.append((main_box, label))
    for inset_map in input["inset_maps"]:
        validate_coordinate(inset_map["map"]["bbox"]["west"])
        validate_coordinate(inset_map["map"]["bbox"]["south"])
        validate_coordinate(inset_map["map"]["bbox"]["east"])
        validate_coordinate(inset_map["map"]["bbox"]["north"])
        inset_box = shapely.box(
            inset_map["map"]["bbox"]["west"],
            inset_map["map"]["bbox"]["south"],
            inset_map["map"]["bbox"]["east"],
            inset_map["map"]["bbox"]["north"],
        )
        layout = inset_map["layout"]
        assert layout["x"] > 0, f"x={layout['x']}"
        assert layout["y"] > 0, f"x={layout['y']}"
        assert layout["scale"] > 0, f"x={layout['scale']}"
        assert main_box.contains(inset_box), f"main={main_box}, inset={inset_box}"
        for point in inset_map["map"]["points"]:
            points.append((inset_box, point))
        for label in inset_map["map"]["labels"]:
            points.append((inset_box, label))
    for box, point in points:
        validate_coordinate(point["lat"])
        validate_coordinate(point["lon"])
        p = shapely.Point(
            point["lon"],
            point["lat"],
        )
        assert box.contains(p), f"box={box}, point={point}"


def download_water_gdf(bbox):
    return osmnx.features_from_bbox(bbox=bbox, tags={"natural": "water"})


def download_grass_gdf(bbox):
    return osmnx.features_from_bbox(
        bbox=bbox,
        tags={
            "natural": [
                "heath",
                "scrub",
                "grassland",
                "shrubbery",
                "tundra",
                "moor",
                "fell",
            ]
        },
    )


def download_forest_gdf(bbox):
    return osmnx.features_from_bbox(
        bbox=bbox, tags={"natural": ["wood", "tree", "tree_row"]}
    )


def download_landuse_gdf(bbox):
    return osmnx.features_from_bbox(
        bbox=bbox,
        tags={"landuse": True, "highway": True},
    )


def plot_basemap(ax, layers):
    for layer in layers:
        gdf = layer["gdf"]
        style = layer["style"]
        gdf.plot(ax=ax, **style)


def plot_scalebar(ax, bbox):
    points = geopandas.GeoSeries(
        [shapely.Point(bbox[3], bbox[0]), shapely.Point(bbox[3] + 1, bbox[0])], crs=4326
    )
    points = points.to_crs(32619)
    distance_meters = points[0].distance(points[1])
    scalebar = ScaleBar(
        dx=distance_meters,
        scale_loc="right",
        box_alpha=0,
        scale_style="geography",
    )
    ax.add_artist(scalebar)


def plot_points(ax, points):
    data = {"names": [], "geometry": []}
    for point in points:
        data["geometry"].append(shapely.Point(point["lon"], point["lat"]))
        data["names"].append(point["name"])
    if len(points) > 0:
        gdf = geopandas.GeoDataFrame(data, crs=4326)
        gdf.plot(ax=ax, color="red", edgecolor="black", zorder=4)
        offset_lon = (ax.get_xlim()[1] - ax.get_xlim()[0]) / 100
        offset_lat = (ax.get_ylim()[1] - ax.get_ylim()[0]) / 100
        for lon, lat, name in zip(gdf.geometry.x, gdf.geometry.y, gdf["names"]):
            text = ax.text(lon + offset_lon, lat + offset_lat, name)
            stroke_effect = [path_effects.withStroke(foreground="w", linewidth=2)]
            text.set_path_effects(stroke_effect)


def plot(input):
    bbox = input["main"]["bbox"]
    bbox = (bbox["north"], bbox["south"], bbox["east"], bbox["west"])

    plt.rcParams["font.size"] = 14

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_facecolor("ivory")
    ax.set_xlim(left=bbox[3], right=bbox[2])
    ax.set_ylim(bottom=bbox[1], top=bbox[0])

    water_gdf = download_water_gdf(bbox)
    grass_gdf = download_grass_gdf(bbox)
    forest_gdf = download_forest_gdf(bbox)
    landuse_gdf = download_landuse_gdf(bbox)

    layers = [
        {
            "gdf": water_gdf,
            "style": {
                "facecolor": "lightskyblue",
                "edgecolor": "deepskyblue",
                "zorder": 3,
            },
        },
        {"gdf": grass_gdf, "style": {"facecolor": "greenyellow", "zorder": 1}},
        {"gdf": forest_gdf, "style": {"facecolor": "palegreen", "zorder": 2}},
        {"gdf": landuse_gdf, "style": {"color": "lightgray", "zorder": 2}},
    ]
    plot_basemap(ax, layers)
    plot_scalebar(ax, bbox)
    plot_points(ax, input["main"]["points"])

    plt.show()


def main():
    args = parse_arguments()
    inputFilePath = Path(args.input)
    assert inputFilePath.is_file()
    input = read_input(inputFilePath)
    validate_input_structure(input)
    add_input_default_values(input)
    convert_input_values(input)
    validate_input_values(input)
    plot(input)


if __name__ == "__main__":
    main()
