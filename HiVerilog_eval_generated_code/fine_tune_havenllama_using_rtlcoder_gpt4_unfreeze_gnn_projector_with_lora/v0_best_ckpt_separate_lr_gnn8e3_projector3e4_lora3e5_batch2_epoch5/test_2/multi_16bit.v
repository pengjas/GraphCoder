module multi_16bit (
    input clk,          
    input rst_n,        
    input start,        
    input [15:0] ain,   
    input [15:0] bin,   
    output [31:0] yout, 
    output done         
);


reg [3:0] shift_count;
reg [0:0] done_flag;

always @(posedge clk or negedge rst_n) begin
    if (~rst_n) begin
        shift_count <= 4'b0000;
        done_flag <= 1'b0;
    end else begin
        if (start) begin
            shift_count <= 4'b0000;
            done_flag <= 1'b0;
        end else begin
            if (shift_count == 4'b1111) begin
                shift_count <= 4'b0000;
                done_flag <= 1'b1;
            end else begin
                shift_count <= shift_count + 4'b0001;
                done_flag <= 1'b0;
            end
        end
    end
end


reg [15:0] a_reg;
reg [15:0] b_reg;

always @(posedge clk or negedge rst_n) begin
    if (~rst_n) begin
        a_reg <= 16'b000000000000000;
        b_reg <= 16'b000000000000000;
    end else begin
        if (start) begin
            a_reg <= ain;
            b_reg <= bin;
        end
    end
end


reg [31:0] product_reg;

always @(posedge clk or negedge rst_n) begin
    if (~rst_n) begin
        product_reg <= 32'b000000000000000000000000000000000;
    end else begin
        if (shift_count == 4'b0000) begin
            product_reg <= {16'b0000000000000000, a_reg};
        end else begin
            if (shift_count[0]) begin
                product_reg <= product_reg + {16'b0000000000000000, b_reg};
            end
        end
    end
end


assign yout = product_reg;
assign done = done_flag;


endmodule