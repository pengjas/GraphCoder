module JC_counter(
   input                clk ,
   input                rst_n,
 
   output reg [63:0]     Q  
);
wire [63:0] Q_next;
    wire [63:0] Q_reg;

    
    reg [63:0] Q_reg_inst;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            Q_reg_inst <= 64'b0;
        else
            Q_reg_inst <= Q_next;
    end

    
    wire [63:0] Q_logic_inst;
    assign Q_logic_inst = Q_reg_inst[63:0];
    assign Q_next = {Q_reg_inst[62:0], Q_logic_inst[0]};

    
    assign Q = Q_reg_inst;
endmodule
\end{code}

\begin{code}
module JC_register(
    input                clk,
    input                rst_n,
    output reg [63:0]   Q
); ASSISTANT:
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            Q <= 64'b0;
        else
            Q <= Q;
    end
end
endmodule

module JC_logic(
    input [63:0]       Q,
    output reg [63:0]   Q_next
); ASSISTANT:
    always @(posedge clk) begin
        Q_next <= {Q[62:0], Q[0]};
    end
endmodule
\end{code}
endmodule