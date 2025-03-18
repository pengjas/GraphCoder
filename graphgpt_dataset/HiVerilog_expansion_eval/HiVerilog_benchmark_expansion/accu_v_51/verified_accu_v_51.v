`timescale 1ns/1ps

module binary_up_counter (
    input               clk,
    input               rst,
    input               en,

    output      [3:0]   count_out,
    output              overflow
);

    wire [3:0] count;
    wire wrap_around;

    // Instantiate the count control module
    count_control u_count_control (
        .clk(clk),
        .rst(rst),
        .en(en),
        .count(count),
        .wrap_around(wrap_around)
    );

    // Instantiate the overflow detection module
    overflow_detection u_overflow_detection (
        .wrap_around(wrap_around),
        .overflow(overflow)
    );

    assign count_out = count;

endmodule

module count_control (
    input               clk,
    input               rst,
    input               en,
    output reg [3:0]   count,
    output              wrap_around
);

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            count <= 0;
        end
        else if (en) begin
            if (count == 4'b1111) begin
                count <= 0;
            end else begin
                count <= count + 1;
            end
        end
    end

    assign wrap_around = (count == 4'b1111) && en;

endmodule

module overflow_detection (
    input               wrap_around,
    output reg          overflow
);

    always @(posedge wrap_around) begin
        overflow <= 1'b1;
    end

    always @(negedge wrap_around) begin
        overflow <= 1'b0;
    end

endmodule
