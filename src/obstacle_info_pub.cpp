#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <derived_object_msgs/ObjectArray.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <shape_msgs/SolidPrimitive.h>

class PointCloudProcessor
{
public:
    PointCloudProcessor()
    {
        ros::NodeHandle nh;
        pointcloud_sub_ = nh.subscribe("/occupied_point", 1, &PointCloudProcessor::pointCloudCallback, this);
        pose_sub_ = nh.subscribe("/vicon/dingo2", 1, &PointCloudProcessor::poseCallback, this);
        object_pub_ = nh.advertise<derived_object_msgs::ObjectArray>("/filtered_objects", 1);
        filtered_points_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/filtered_points", 1);
    }

private:
    ros::Subscriber pointcloud_sub_;
    ros::Subscriber pose_sub_;
    ros::Publisher object_pub_;
    ros::Publisher filtered_points_pub_;
    geometry_msgs::PoseStamped current_pose_;

    void poseCallback(const geometry_msgs::PoseStamped::ConstPtr &msg)
    {
        current_pose_ = *msg;
    }

    void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr &cloud_msg)
    {
        pcl::PointCloud<pcl::PointXYZ> cloud;
        pcl::fromROSMsg(*cloud_msg, cloud);
        derived_object_msgs::ObjectArray object_array;
        object_array.header = cloud_msg->header;

        pcl::PointCloud<pcl::PointXYZ> filtered_cloud;

        shape_msgs::SolidPrimitive sphere;
        sphere.type = shape_msgs::SolidPrimitive::SPHERE;
        sphere.dimensions.resize(1);
        sphere.dimensions[0] = 0.075;

        int count = 0;
        int max_count = 100;
        for (const auto &point : cloud.points)
        {
            double dx = point.x - current_pose_.pose.position.x;
            double dy = point.y - current_pose_.pose.position.y;
            double distance = std::sqrt(dx * dx + dy * dy);

            if (point.z > 0.15 && distance <= 3.0)
            {
                derived_object_msgs::Object obj;
                obj.pose.position.x = point.x;
                obj.pose.position.y = point.y;
                obj.pose.position.z = point.z;
                obj.shape = sphere;

                object_array.objects.push_back(obj);
                filtered_cloud.push_back(point);
                count++;
                if (count >= max_count)
                    break;
            }
        }

        int num_objects = object_array.objects.size();

        while (num_objects < max_count)
        {
            derived_object_msgs::Object obj;
            obj.pose.position.x = 100;
            obj.pose.position.y = 100;
            obj.pose.position.z = 100;
            obj.shape = sphere;
            object_array.objects.push_back(obj);
            num_objects++;
        }

        object_pub_.publish(object_array);

        sensor_msgs::PointCloud2 filtered_cloud_msg;
        pcl::toROSMsg(filtered_cloud, filtered_cloud_msg);
        filtered_cloud_msg.header = cloud_msg->header;
        filtered_points_pub_.publish(filtered_cloud_msg);
        ROS_INFO_THROTTLE(1, "Publishing %d obstacles. Valid: %d.", num_objects, count);
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "pointcloud_processor");
    PointCloudProcessor processor;

    ROS_INFO("Obstacle info publisher initialized.");
    
    ros::spin();
    return 0;
}
