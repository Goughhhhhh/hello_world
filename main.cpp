#include <rosefusion.h>
#include <DataReader.h>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <ctime>
#include <fstream>
#include <pangolin/pangolin.h>


// demo
int main(int argc,char* argv[]){

    std::cout<<"Read configure file\n";
    const std::string camera_file(argv[1]);
    const std::string data_file(argv[2]);
    const std::string controller_file(argv[3]);

    std::cout<<"Init configuration\n";
    printf("%s\n",camera_file.c_str());
    printf("%s\n",data_file.c_str());
    printf("%s\n",controller_file.c_str());

    const rosefusion::CameraParameters camera_config(camera_file);
    const rosefusion::DataConfiguration data_config(data_file);
    const rosefusion::ControllerConfiguration controller_config(controller_file);

    pangolin::View color_cam;
    pangolin::View shaded_cam; 
    pangolin::View depth_cam; 

    pangolin::GlTexture imageTexture;
    pangolin::GlTexture shadTexture;
    pangolin::GlTexture depthTexture;

    if (controller_config.render_surface){

        pangolin::CreateWindowAndBind("Main",2880,1440);

        color_cam = pangolin::Display("color_cam")
            .SetAspect((float)camera_config.image_width/(float)camera_config.image_height);
        shaded_cam = pangolin::Display("shaded_cam")
            .SetAspect((float)camera_config.image_width/(float)camera_config.image_height);
        depth_cam = pangolin::Display("depth_cam")
            .SetAspect((float)camera_config.image_width/(float)camera_config.image_height);

        pangolin::Display("window")
            .SetBounds(0.0, 1.0, 0.0, 1.0 )
            .SetLayout(pangolin::LayoutEqual)
            .AddDisplay(shaded_cam)
            .AddDisplay(color_cam)
            .AddDisplay(depth_cam);


    
        imageTexture=pangolin::GlTexture(camera_config.image_width,camera_config.image_height,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);
        shadTexture=pangolin::GlTexture(camera_config.image_width,camera_config.image_height,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);
        depthTexture=pangolin::GlTexture(camera_config.image_width,camera_config.image_height,GL_LUMINANCE,false,0,GL_LUMINANCE,GL_UNSIGNED_BYTE);
    }
    cv::Mat shaded_img(camera_config.image_height, camera_config.image_width,CV_8UC3);

    rosefusion::Pipeline pipeline { camera_config, data_config, controller_config }; // 用三个参数初始化一个rosefusion对象

    clock_t time_stt=clock( ); //时间统计
    cv::Mat color_img;
    cv::Mat depth_map;  
    int n_imgs=0; //用于当前帧计数
    
    std::cout<<"Read seq file: "<<data_config.seq_file<<"\n";

    DataReader d_reader(data_config.seq_file,false);  //初始化DataReader对象的时候会初始化当前帧序号currentFrame=0;

    while( d_reader.hasMore()){ // currentFrame<numFrames 还有帧没有被处理完


        printf("n:%d\n",n_imgs);

        d_reader.getNextFrame(color_img,depth_map); //currentFrame++;
        bool success = pipeline.process_frame(depth_map, color_img,shaded_img); ////////////////////////////////////////////////rosefusion主算法函数在此

        if (!success){
            std::cout << "Frame could not be processed" << std::endl;
        }


        if (controller_config.render_surface){
            glClear(GL_COLOR_BUFFER_BIT);

            color_cam.Activate(); //显示设置
            imageTexture.Upload(color_img.data,GL_BGR,GL_UNSIGNED_BYTE);
            imageTexture.RenderToViewportFlipY();
            depth_cam.Activate();

            depth_map.convertTo(depth_map,CV_8U,256/5000.0);
            depthTexture.Upload(depth_map.data,GL_LUMINANCE,GL_UNSIGNED_BYTE);
            depthTexture.RenderToViewportFlipY();

            if (success){
                shaded_cam.Activate();
                shadTexture.Upload(shaded_img.data,GL_BGR,GL_UNSIGNED_BYTE);
                shadTexture.RenderToViewportFlipY();
            }
            pangolin::FinishFrame();

        }

        n_imgs++;

    }


    std::cout <<"time per frame="<<1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC/n_imgs<<"ms"<<std::endl; // 统计每一帧的平均处理时间

    if (controller_config.save_trajectory){
        pipeline.get_poses(); // 保存一个.text的轨迹文件
    }

    if (controller_config.save_scene){
        auto points = pipeline.extract_pointcloud(); // ROSEFusion只重建有三维点云地图
        rosefusion::export_ply(data_config.result_path+data_config.seq_name+"_points.ply",points);
    }


    return 0;
}
