#include <memory>
#include <verilated.h>
#include <opencv2/opencv.hpp>
#include "Vtb_verilator.h"
#include "jelly/simulator/Manager.h"
#include "jelly/simulator/VerilatorNode.h"
#include "jelly/simulator/Axi4sImageLoadNode.h"
#include "jelly/simulator/Axi4sImageDumpNode.h"


namespace jsim = jelly::simulator;


#if VM_TRACE
#include <verilated_fst_c.h> 
#include <verilated_vcd_c.h> 
#endif


// 画像の前処理
cv::Mat PreProcImage(cv::Mat img, int frame_num)
{
    if ( img.empty() ) { return img; }

    // 回転
    auto center = cv::Point(img.cols/2, img.rows/2);
    auto trans  = cv::getRotationMatrix2D(center, frame_num*10, 1);
    cv::Mat img2;
    cv::warpAffine(img, img2, trans, img.size());
    return img2;
}


int main(int argc, char** argv)
{
    auto contextp = std::make_shared<VerilatedContext>();
    contextp->debug(0);
    contextp->randReset(2);
    contextp->commandArgs(argc, argv);
    
    const auto top = std::make_shared<Vtb_verilator>(contextp.get(), "top");
    
    
    jsim::trace_ptr_t tfp = nullptr;
#if VM_TRACE
    contextp->traceEverOn(true);

    tfp = std::make_shared<jsim::trace_t>();
    top->trace(tfp.get(), 100);
    tfp->open("tb_verilator" TRACE_EXT);
#endif

    auto mng = jsim::Manager::Create();

    mng->AddNode(jsim::ClockNode_Create(&top->clk, 5.0/2));
    mng->AddNode(jsim::ResetNode_Create(&top->reset, 100));
    mng->AddNode(jsim::VerilatorNode_Create(top, tfp));

/*
    jsim::Axi4sVideo axi4s_src =
            {
                &top->aresetn,
                &top->aclk,
                &top->s_axi4s_src_tuser,
                &top->s_axi4s_src_tlast,
                &top->s_axi4s_src_tdata,
                &top->s_axi4s_src_tvalid,
                &top->s_axi4s_src_tready
            };

    jsim::Axi4sVideo axi4s_dst =
            {
                &top->aresetn,
                &top->aclk,
                &top->m_axi4s_dst_tuser,
                &top->m_axi4s_dst_tlast,
                &top->m_axi4s_dst_tdata,
                &top->m_axi4s_dst_tvalid,
                &top->m_axi4s_dst_tready
            };

    jsim::Axi4sVideo axi4s_angle =
            {
                &top->aresetn,
                &top->aclk,
                &top->m_axi4s_angle_tuser,
                &top->m_axi4s_angle_tlast,
                &top->m_axi4s_angle_tdata,
                &top->m_axi4s_angle_tvalid,
                (int*)nullptr
            };

    std::string s;
    auto image_src_load = jsim::Axi4sImageLoadNode_Create(axi4s_src, "../BOAT.bmp", jsim::fmt_gray);
    auto image_dst_dump = jsim::Axi4sImageDumpNode_Create(axi4s_dst, "img_%04d.png", jsim::fmt_gray, 256, 256);
    auto image_angle_dump = jsim::Axi4sImageDumpNode_Create(axi4s_angle, "angle_%04d.png", jsim::fmt_color, 256, 256);

    image_src_load->SetBlankX(64);
    image_src_load->SetBlankY((256 + 64) * 8);
    image_src_load->SetRandomWait(0.0);
    image_dst_dump->SetRandomWait(0.0);

    image_angle_dump->SetFrameLimit(2);
    image_dst_dump->SetFrameLimit(2);//, true); // 2フレーム処理したら終了するように設定
    

    mng->AddNode(image_src_load);
    mng->AddNode(image_dst_dump);
    mng->AddNode(image_angle_dump);

    image_src_load->SetImagePreProc(PreProcImage);
    image_src_load->SetImageShow("sorce image");
    image_angle_dump->SetImageShow("canny angle");
    image_dst_dump->SetImageShow("canny edge");
    mng->SetThreadEnable(true);

    mng->SetControlCvWindow("Simulation", 0x1b);
*/

    mng->Run(10000000);
//    mng->Run();

#if VM_TRACE
    tfp->close();
#endif

#if VM_COVERAGE
    contextp->coveragep()->write("coverage.dat");
#endif

    return 0;
}


// end of file
