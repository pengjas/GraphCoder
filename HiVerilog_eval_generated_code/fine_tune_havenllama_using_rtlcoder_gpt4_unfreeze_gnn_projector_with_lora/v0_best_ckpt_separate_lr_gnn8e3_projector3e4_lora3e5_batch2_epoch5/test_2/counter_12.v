module counter_12 (
    input rst_n,
    input clk,
    input valid_count,
    output reg [3:0] out
);

    reg [3:0] count;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count <= 4'b0000;
        end else if (valid_count) begin
            if (count == 4'd11) begin
                count <= 4'b0000;
            end else begin
                count <= count + 1'b1;
            end
        end
    end

    always @(posedge clk) begin
        if (valid_count) begin
            out <= count;
        end
    end

endmodule

module output_register (
    input rst_n,
    input clk,
    input valid_count,
    input [3:0] count,
    output reg [3:0] out
);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out <= 4'b0000;
        end else if (valid_count) begin
            out <= count;
        end
    end

endmodule

module counter_12 (
    input rst_n,
    input clk,
    input valid_count,
    output reg [3:0] out
);

    wire [3:0] count;

    counter_logic u_counter_logic (
        .rst_n(rst_n),
        .clk(clk),
        .valid_count(valid_count),
        .count(count)
    );

    output_register u_output_register (
        .rst_n(rst_n),
        .clk(clk),
        .valid_count(valid_count),
        .count(count),
        .out(out)
    );

endmodule