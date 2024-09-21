/****
 * @file object_info_handler.h
 * @brief This file contains the declaration of the class ObjectInfoHandler, which is used to read the object information csv file and get thre required data in data_base.h
 * @author Clarence Chen
 * @version 0.1
 * 
 * @copyright Copyright (c) 2023
 * ***/

#pragma once

#include "data_base.h"
#include <fstream>
#include <iostream>


class ObjectInfoHandler
{
public:
    ObjectInfoHandler() = default;
    ~ObjectInfoHandler() = default;

    /**
     * @brief Read the object information csv file and store the data in the object_info_map_
     * 
     * @param file_path The path of the object information csv file
     */
    void readObjectInfo(const std::string &file_path)
    {
        // Check if the file exists and the suffix is csv
        if (file_path.substr(file_path.find_last_of(".") + 1) != "csv")
        {
            std::cerr << "The file path is not a csv file. Will use default settings in data_base.h" << std::endl;
            return;
        }

        // Read the csv file
        std::vector<CsvRow> data = readCsv(file_path);

        std::unordered_map<std::string, int> label_id_map;
        std::unordered_map<int, std::string> label_id_map_reversed;
        std::unordered_set<int> movable_object_label_ids_set;
        std::unordered_map<std::string, int> label_id_map_static;
        std::unordered_map<int, std::string> label_id_map_static_reversed;
        std::unordered_map<std::string, int> label_to_instance_id_map;
        std::unordered_map<int, std::string> instance_id_to_label_map;
        std::unordered_map<int, cv::Vec3b> label_color_map;

        int max_movable_object_instance_id = 65535;

        for(const auto &row : data)
        {
            label_id_map[row.label] = row.label_id;
            label_id_map_reversed[row.label_id] = row.label;

            label_color_map[row.label_id] = cv::Vec3b(row.color_b, row.color_g, row.color_r);

            if(row.instance_id > 0) // Static object. Instance id required
            {
                label_id_map_static[row.label] = row.label_id;
                label_id_map_static_reversed[row.label_id] = row.label;
                label_to_instance_id_map[row.label] = row.instance_id;
                instance_id_to_label_map[row.instance_id] = row.label;

                int movable_object_instance_id_temp = row.instance_id - 1;
                if(movable_object_instance_id_temp < max_movable_object_instance_id)
                {
                    max_movable_object_instance_id = movable_object_instance_id_temp;
                }
            }else{ // -1. Movable object
                movable_object_label_ids_set.insert(row.label_id);
            }
        }

        // Update the global variables in data_base.h
        g_label_id_map_default = label_id_map;
        g_label_id_map_reversed = label_id_map_reversed;

        g_movable_object_label_ids_set = movable_object_label_ids_set;
        g_label_id_map_static = label_id_map_static;

        g_label_id_map_static_reversed = label_id_map_static_reversed;
        g_label_to_instance_id_map_default = label_to_instance_id_map;
        g_max_movable_object_instance_id = max_movable_object_instance_id;

        g_instance_id_to_label_map_default = instance_id_to_label_map;
        g_label_color_map_default = label_color_map;

        std::cout << "Object information csv file read successfully. File: " << file_path << std::endl;
        std::cout << " g_max_movable_object_instance_id = " << g_max_movable_object_instance_id << std::endl;
    }
    

private:
    // Define a structure to hold each row's data
    struct CsvRow {
        int label_id;
        std::string label;
        int instance_id;
        int color_b;
        int color_g;
        int color_r;
    };

    // Function to parse a line into a CsvRow
    CsvRow parseLine(const std::string& line) {
        std::stringstream ss(line);
        std::string item;
        CsvRow row;

        // Parse each field and store in the corresponding structure member
        std::getline(ss, item, ',');
        row.label_id = std::stoi(item);

        std::getline(ss, item, ',');
        row.label = item;

        std::getline(ss, item, ',');
        row.instance_id = std::stoi(item);

        std::getline(ss, item, ',');
        row.color_b = std::stoi(item);

        std::getline(ss, item, ',');
        row.color_g = std::stoi(item);

        std::getline(ss, item, ',');
        row.color_r = std::stoi(item);

        return row;
    }

    // Function to read the CSV file and store the data in a vector of CsvRow
    std::vector<CsvRow> readCsv(const std::string& filename) {
        std::ifstream file(filename);
        std::vector<CsvRow> data;
        std::string line;

        // Skip the header line
        std::getline(file, line);

        // Read each line and parse it into a CsvRow, then store it in the vector
        while (std::getline(file, line)) {
            CsvRow row = parseLine(line);
            data.push_back(row);
        }

        return data;
    }
    

};
